import argparse
import os
import math
import time
import torch
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.load_pt_dataset import PretrainDataset
from utils.load_sft_dataset import SFTSQLGenerationDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate.utils import set_seed
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR, SinusoidalLR

from torch.nn import functional as F
from torch.nn import KLDivLoss, MSELoss

"""
Training LLM using Huggingface Accelerate + Deepspeed.
"""


def parse_option():
    parser = argparse.ArgumentParser()

    # global args
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="batch size per gpu device.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=8192,
        help="block size, i.e., the length of training sequences.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, default="bigcode/starcoder"
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="5e-5 for pre-training, 5e-6 for fine-tuning.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
        help="ratio of total training steps used for a linear warmup from 0 to max lr.",
    )
    parser.add_argument("--checkpointing_steps", type=int, default=300)
    parser.add_argument("--tensorboard_log_dir", type=str, default="./train_logs")
    parser.add_argument("--mode", type=str, default="pt")
    parser.add_argument("--output_ckpt_dir", type=str, default="./ckpts")
    parser.add_argument(
        "--save_all_states",
        action="store_true",
        default=False,
        help="whether to save states of model, optimizer, and lr scheduler for resuming training, otherwise only model states are saved.",
    )

    # args for pre-training
    parser.add_argument("--pt_data_dir", type=str, default="./data/corpus.bin")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="resuming pre-training from a checkpoint",
    )
    parser.add_argument("--resume_tag", type=str, default=None)

    # args for supervised fine-tuning
    parser.add_argument(
        "--text2sql_data_dir", type=str, default="./data/sft_train_text2sql.json"
    )
    parser.add_argument("--table_num", type=int, default=6)
    parser.add_argument("--column_num", type=int, default=10)

    # kd args
    parser.add_argument("--teacher_model", type=str, default="seeklhy/codes-3b-spider")
    parser.add_argument("--clm_beta", type=float, default=0.5)
    parser.add_argument("--clm_start_beta", type=float, default=0.5)
    parser.add_argument("--clm_beta_scheduler", type=str, default="const")
    parser.add_argument("--kd_loss", type=str, default="rev_kl")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--mixin_alpha", type=float, default=0.2)

    # kd mix-in scheduler
    parser.add_argument("--alpha_warmup_ratio", type=float, default=1)
    parser.add_argument("--alpha_scheduler", type=str, default="step")
    parser.add_argument("--min_alpha", type=float, default=0.01)
    parser.add_argument("--alpha_step_size", type=int, default=10)
    parser.add_argument("--start_alpha", type=float, default=0.5)
    parser.add_argument("--imp_weight", type=int, default=1)
    parser.add_argument("--direction", type=str, default="normal")

    opt = parser.parse_args()

    return opt


def checkpoint_model_optimizer_scheduler(
    checkpoint_folder, model, last_global_step, lr_scheduler, accelerator
):
    """
    Utility function for checkpointing model + optimizer dictionaries
    The main purpose for this is to be able to resume training from that instant again
    """
    checkpoint_state_dict = {
        "last_global_step": last_global_step,
    }

    accelerator.print("==> saving model and optimizer <==")
    model.save_checkpoint(checkpoint_folder, last_global_step, checkpoint_state_dict)

    accelerator.print("==> saving lr scheduler <==")
    accelerator.save(
        lr_scheduler.state_dict(),
        os.path.join(checkpoint_folder, str(last_global_step), "scheduler.pt"),
    )

    print(
        f"checkpointing: checkpoint_folder={checkpoint_folder}, ckpt_id={last_global_step}"
    )
    return


def resume_model_and_optimizer(model, load_dir, tag):
    """
    Utility function for checkpointing model + optimizer dictionaries
    The main purpose for this is to be able to resume training from that instant again
    """
    _, checkpoint_state_dict = model.load_checkpoint(
        load_dir, tag=tag, load_optimizer_states=True
    )

    last_global_step = checkpoint_state_dict["last_global_step"]
    del checkpoint_state_dict

    return last_global_step


def checkpoint_model(accelerator, model, tokenizer, output_ckpt_dir, last_global_step):
    """
    Utility fuction for only checkpointing the model dictionary (i.e., only model parameters)
    """
    ckpt_path = os.path.join(output_ckpt_dir, "ckpt-{}".format(last_global_step))
    accelerator.print("checkpointing model state dict at {}".format(ckpt_path))
    unwrapped_model = accelerator.unwrap_model(model)
    # TODO: currently, there is a small bug that saves a full checkpoint data for each shard when enable zero1 and 2.
    # See https://github.com/microsoft/DeepSpeed/issues/3303. solution: waiting upgrade of accelerate and deepspeed
    unwrapped_model.save_pretrained(
        ckpt_path,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
        max_shard_size="100GB",
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(ckpt_path)

    return


def sanity_check(input, target, tokenizer):
    print("Start Sanity Check -------->")
    for t, m in zip(input[:-1], target[1:]):
        decoded = tokenizer.decode([t])
        print("%20s: %6d -> %6d" % (repr(decoded), t, m))
    print("<-------- End Sanity Check")


def check_convergence(accelerator, losses, patience, threshold=1e-4):
    if len(losses) < patience + 1:
        return False

    recent_losses = losses[-(patience + 1) :]
    improvements = [
        recent_losses[i] - recent_losses[i - 1] for i in range(1, len(recent_losses))
    ]

    accelerator.print("Recent losses: {}".format(recent_losses))
    accelerator.print("Improvements: {}".format(improvements))

    if all(abs(improvement) < threshold for improvement in improvements):
        accelerator.print(
            "Convergence criteria met. Improvements less than threshold {} for {} epochs.".format(
                threshold, patience
            )
        )
        return True

    accelerator.print("Convergence criteria not met. Continuing training.")
    return False


def _generate(model, tokenizer, input_ids, attention_mask):
    return model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=8192,
        pad_token_id=tokenizer.eos_token_id,
        num_beams=4,
        do_sample=True,
        temperature=1,
    )


def mixin_logits(student_logits, teacher_logits, mixin_alpha):
    mixin = (1 - mixin_alpha) * student_logits + mixin_alpha * teacher_logits
    return mixin


def mixin_alpha_scheduler(
    step,
    total_steps,
    step_size=3,
    schedule_type="cos",
    min_alpha=0,
    start_alpha=0.2,
    direction="normal",
):
    if schedule_type == "linear":
        alpha = start_alpha * ((total_steps - step) / total_steps)
    elif schedule_type == "exponential":
        k = -np.log(0.01) / total_steps
        alpha = start_alpha * np.exp(-k * step)
    elif schedule_type == "step":
        N = int(total_steps / step_size)
        alpha = start_alpha * (int((total_steps - step) / step_size) / N)
    elif schedule_type == "cos":
        alpha = (start_alpha / 2) * (1 + np.cos(step_size * np.pi * step / total_steps))
    # elif schedule_type == "cos2":
    #     cycle_step = step % (total_steps // 2)
    #     alpha = start_alpha * (1 + np.cos(np.pi * cycle_step / (total_steps // 2))) / 2
    elif schedule_type == "const":
        alpha = start_alpha
    else:
        raise ValueError("Unknown schedule_type: {}".format(schedule_type))

    if direction == "inverse":
        return start_alpha - alpha
    return alpha


def compute_kd_loss(
    student_logits, teacher_logits, kd_loss_type, temperature, mixin_alpha, iww=-1
):
    m_student_logits = mixin_logits(student_logits, teacher_logits, mixin_alpha)

    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    m_student_logits = m_student_logits / temperature

    assert m_student_logits.shape == student_logits.shape == teacher_logits.shape

    if kd_loss_type == "fkl":
        if mixin_alpha != 0:
            if iww == -1:
                imp_weight = 1
            else:
                imp_weight = torch.mean(
                    F.softmax(student_logits, dim=-1)
                    / F.softmax(m_student_logits, dim=-1)
                )
            return (
                imp_weight
                * KLDivLoss(reduction="batchmean")(
                    F.log_softmax(m_student_logits, dim=-1),
                    F.softmax(teacher_logits, dim=-1),
                )
                * (temperature**2),
                imp_weight,
            )

        else:
            return (
                KLDivLoss(reduction="batchmean")(
                    F.log_softmax(student_logits, dim=-1),
                    F.softmax(teacher_logits, dim=-1),
                )
                * (temperature**2),
                1,
            )

    elif kd_loss_type == "revkl":
        if mixin_alpha != 0:
            if iww == -1:
                imp_weight = 1
            else:
                imp_weight = torch.mean(
                    F.softmax(student_logits, dim=-1)
                    / F.softmax(m_student_logits, dim=-1)
                )
            return (
                imp_weight
                * KLDivLoss(reduction="batchmean")(
                    F.log_softmax(teacher_logits, dim=-1),
                    F.softmax(m_student_logits, dim=-1),
                )
                * (temperature**2),
                imp_weight,
            )
        else:
            return (
                KLDivLoss(reduction="batchmean")(
                    F.log_softmax(teacher_logits, dim=-1),
                    F.softmax(student_logits, dim=-1),
                )
                * (temperature**2),
                1,
            )

    elif kd_loss_type == "ce":
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        log_student_probs = F.log_softmax(student_logits, dim=-1)
        return -(teacher_probs * log_student_probs).sum(dim=-1).mean(), 1

    elif kd_loss_type in ["mse", "pmse"]:
        if kd_loss_type == "p_mse":
            p_student = F.softmax(student_logits, dim=-1)
            p_teacher = F.softmax(teacher_logits, dim=-1)
            return MSELoss()(p_student, p_teacher), 1
        elif kd_loss_type == "mse":
            return (
                MSELoss()(student_logits * temperature, teacher_logits * temperature),
                1,
            )

    elif kd_loss_type == "js":
        m = 0.5 * (
            F.softmax(student_logits, dim=-1) + F.softmax(teacher_logits, dim=-1)
        )
        return (
            0.5
            * (
                KLDivLoss(reduction="batchmean")(
                    F.log_softmax(student_logits, dim=-1), m
                )
                + KLDivLoss(reduction="batchmean")(
                    F.log_softmax(teacher_logits, dim=-1), m
                )
            ),
            1,
        )
    elif kd_loss_type == "mixkl":
        # term1 = w * (1 - alpha) * q * np.log((1 - alpha) * q / p)
        # term2 = -w * alpha * p * np.log(p / ((1 - alpha) * q))

        imp_weight = torch.mean(
            F.softmax(student_logits, dim=-1) / F.softmax(m_student_logits, dim=-1)
        )
        term1 = imp_weight * KLDivLoss(reduction="batchmean")(
            F.log_softmax(teacher_logits, dim=-1),
            F.softmax((1 - mixin_alpha) * student_logits, dim=-1),
        )
        term2 = (
            -imp_weight
            * mixin_alpha
            * KLDivLoss(reduction="batchmean")(
                F.log_softmax((1 - mixin_alpha) * student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1),
            )
        )
        return (term1 + term2), 1
    else:
        raise ValueError(
            "kd_loss should be in ['revkl', 'fkl', 'js', 'ce', 'mse', 'pmse']."
        )


def calculate_loss(outputs, teacher_outputs, clm_loss, opt):
    if opt.kd_loss or opt.clm_beta != 1:
        # student_logits, teacher_logits, kd_loss_type, temperature, mixin_alpha, iww
        kd_loss, imp_weight = compute_kd_loss(
            student_logits=outputs.logits,
            teacher_logits=teacher_outputs.logits,
            kd_loss_type=opt.kd_loss,
            temperature=opt.temperature,
            mixin_alpha=opt.mixin_alpha,
            iww=opt.imp_weight,
        )
        return opt.clm_beta * clm_loss + (1 - opt.clm_beta) * kd_loss, imp_weight
    else:
        return clm_loss, -1


def train(opt):
    set_seed(opt.seed)

    writer = SummaryWriter(opt.tensorboard_log_dir)
    accelerator = Accelerator()

    print("accelerator.is_main_process:", accelerator.is_main_process)
    print("accelerator.device:", accelerator.device)

    total_batch_size = (
        opt.per_device_train_batch_size
        * accelerator.num_processes
        * accelerator.gradient_accumulation_steps
    )

    accelerator.print(opt)
    accelerator.print("tokens per batch:", total_batch_size * opt.block_size)
    accelerator.print("sequences per batch:", total_batch_size)
    accelerator.print("using LLM from:", opt.pretrained_model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(
        opt.pretrained_model_name_or_path,
        token="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",
    )
    model = AutoModelForCausalLM.from_pretrained(
        opt.pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16,
        token="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",
        attn_implementation="flash_attention_2",
    )

    teacher_model = AutoModelForCausalLM.from_pretrained(
        opt.teacher_model,
        torch_dtype=torch.bfloat16,
        token="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",
        attn_implementation="flash_attention_2",
    )

    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id
    teacher_model.config.pad_token_id = tokenizer.eos_token_id

    # enable gradient checkpointing to save GPU memory, but this action would slowdown the training speed 20-30%
    model.gradient_checkpointing_enable()

    if opt.mode == "pt":
        dataset = PretrainDataset(opt.pt_data_dir, opt.block_size)
        if accelerator.is_main_process:
            sanity_check(dataset[0]["input_ids"], dataset[0]["labels"], tokenizer)
    elif opt.mode == "sft":
        dataset = SFTSQLGenerationDataset(
            opt.text2sql_data_dir,
            tokenizer,
            opt.block_size,
            "train",
            opt.table_num,
            opt.column_num,
            None,
        )
    elif opt.mode == "seqkd":  # sample response/logits directly from teacher model
        dataset = SFTSQLGenerationDataset(
            text2sql_data_dir=opt.text2sql_data_dir,
            tokenizer=tokenizer,
            max_tokens=opt.block_size,
            mode="seqkd",
            table_num=opt.table_num,
            column_num=opt.column_num,
            sic_path=None,
            teacher_model=teacher_model,
        )
    else:
        raise ValueError("opt.mode should be in [pt, sft, seqkd].")
    dataloader = DataLoader(
        dataset,
        batch_size=opt.per_device_train_batch_size,
        num_workers=32,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )

    num_total_batches = math.ceil(
        opt.epochs * math.ceil(len(dataset) / total_batch_size)
    )  # number of total batches
    optimizer = AdamW(
        model.parameters(), lr=opt.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
    )

    num_warm_up_batches = max(int(num_total_batches * opt.warmup_ratio), 1)
    accelerator.print("num_warm_up_batches:", num_warm_up_batches)

    lr_scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=num_warm_up_batches * accelerator.num_processes,
        max_epochs=num_total_batches * accelerator.num_processes,
        warmup_start_lr=0.0,
        eta_min=0.01 * opt.lr,
    )

    # lr_scheduler = SinusoidalLR(
    #     optimizer=optimizer,
    #     max_epochs=num_total_batches * accelerator.num_processes,
    #     step=3,
    #     eta_min=0.01 * opt.lr,
    # )

    optimizer, model, dataloader, lr_scheduler = accelerator.prepare(
        optimizer, model, dataloader, lr_scheduler
    )

    teacher_model = teacher_model.to(accelerator.device)
    print(type(optimizer))
    print(type(model))
    print(type(dataloader))
    print(type(lr_scheduler))

    accumulation_loss = 0
    global_completed_steps = 0

    model.train()
    teacher_model.eval()

    # resume pre-training if opt.resume_from_checkpoint is not None
    if opt.mode == "pt" and opt.resume_from_checkpoint:
        # resume model and optimizer states
        global_completed_steps = resume_model_and_optimizer(
            model, opt.resume_from_checkpoint, opt.resume_tag
        )

        resume_epoch = (
            global_completed_steps
            * accelerator.gradient_accumulation_steps
            // len(dataloader)
        )
        resume_batch_idx = (
            global_completed_steps
            * accelerator.gradient_accumulation_steps
            % len(dataloader)
        )

        accelerator.print("resume epoch:", resume_epoch)
        accelerator.print("resume batch index:", resume_batch_idx)
        accelerator.print(
            "resume training from {}".format(
                os.path.join(opt.resume_from_checkpoint, opt.resume_tag)
            )
        )

        # resume lr scheduler
        lr_scheduler.load_state_dict(
            torch.load(
                os.path.join(opt.resume_from_checkpoint, opt.resume_tag, "scheduler.pt")
            )
        )
        accelerator.print("lr scheduler state dict:", lr_scheduler.state_dict())

    st = time.time()
    losses = []  # Initialize a list to keep track of loss values
    for epoch in range(opt.epochs):
        if opt.mode == "pt" and opt.resume_from_checkpoint and resume_epoch > epoch:
            accelerator.print("skip {}-th epoch".format(epoch))
            continue
        accelerator.print("Start training epoch:", epoch + 1)
        for batch_idx, batch in enumerate(dataloader):
            if (
                opt.mode == "pt"
                and opt.resume_from_checkpoint
                and resume_batch_idx > batch_idx
            ):
                accelerator.print("skip {}-th batch".format(batch_idx))
                continue

            # `accelerator.accumulate(model)` aims to set right `sync_gradients` state based on the recorded training steps
            with accelerator.accumulate(model):
                # --
                outputs = model(**batch)  # student logits
                if opt.clm_beta != 1:
                    with torch.no_grad():  # calculate teacher logits
                        teacher_outputs = teacher_model(**batch)

                clm_loss = outputs.loss  # CE with labels
                # --

                mixin_scheduler_config = {
                    "step": global_completed_steps,
                    "total_steps": (
                        num_total_batches
                        * accelerator.num_processes
                        * opt.alpha_warmup_ratio
                    ),
                    "step_size": opt.alpha_step_size,
                    "schedule_type": opt.alpha_scheduler,
                    "min_alpha": opt.min_alpha,
                    "start_alpha": opt.start_alpha,
                    "direction": opt.direction,
                }
                # if opt.mixin_alpha != 0:
                opt.mixin_alpha = mixin_alpha_scheduler(**mixin_scheduler_config)

                clm_beta_config = mixin_scheduler_config.copy()
                clm_beta_config["schedule_type"] = opt.clm_beta_scheduler
                clm_beta_config["start_alpha"] = opt.clm_start_beta
                clm_beta_config["direction"] = "normal"

                opt.clm_beta = mixin_alpha_scheduler(**clm_beta_config)

                # --
                loss, imp_weight = calculate_loss(
                    outputs, teacher_outputs, clm_loss, opt
                )
                accumulation_loss += loss.detach().float()

                accelerator.backward(loss)
                optimizer.step()
                # --

                lr_scheduler.step()
                optimizer.zero_grad()

            # 'accelerator.sync_gradients' checks if the accelerator has performed an optimization step on the `total_batch_size` examples
            if accelerator.sync_gradients:
                global_completed_steps += 1
                # --
                if opt.kd_loss == "mse":
                    loss = (
                        accumulation_loss
                        * 1e5
                        / accelerator.gradient_accumulation_steps
                    )
                else:
                    loss = accumulation_loss / accelerator.gradient_accumulation_steps

                accelerator.print(
                    "GPU 0, step {}, loss {:.5f}, mix-in {:.4f}, imp_weight {:.4f}".format(
                        global_completed_steps, loss, opt.mixin_alpha, imp_weight
                    )
                )
                # --

                accelerator.print(
                    "GPU 0, step {}, lr state dict:".format(global_completed_steps),
                    lr_scheduler.state_dict(),
                )
                accelerator.print(time.time() - st)
                st = time.time()

                writer.add_scalar(
                    "train-loss/gpu-{}".format(accelerator.process_index),
                    accumulation_loss / accelerator.gradient_accumulation_steps,
                    global_completed_steps,
                )
                writer.add_scalar(
                    "learning-rate/gpu-{}".format(accelerator.process_index),
                    lr_scheduler.get_last_lr()[0],
                    global_completed_steps,
                )
                writer.add_scalar(
                    "mix-in/gpu-{}".format(accelerator.process_index),
                    opt.mixin_alpha,
                    global_completed_steps,
                )
                writer.add_scalar(
                    "clm_beta/gpu-{}".format(accelerator.process_index),
                    opt.clm_beta,
                    global_completed_steps,
                )

                # --
                writer.add_scalar(
                    "imp-weight/gpu-{}".format(accelerator.process_index),
                    imp_weight,
                    global_completed_steps,
                )

                # reset accumulation_loss to 0
                losses.append(
                    accumulation_loss.item() / accelerator.gradient_accumulation_steps
                )
                accumulation_loss = 0
                # save checkpoints for each checkpointing_steps total batch size
                if global_completed_steps % opt.checkpointing_steps == 0:
                    accelerator.wait_for_everyone()
                    checkpoint_model(
                        accelerator,
                        model,
                        tokenizer,
                        opt.output_ckpt_dir,
                        global_completed_steps,
                    )
                    if opt.save_all_states:
                        checkpoint_model_optimizer_scheduler(
                            opt.output_ckpt_dir,
                            model,
                            global_completed_steps,
                            lr_scheduler,
                            accelerator,
                        )
                # --
        # if opt.mode == "pt" or (opt.mode == "sft" and (epoch+1)%2 == 0):
        # --
        accelerator.print("in the end of an epoch, save a checkpoint")
        accelerator.wait_for_everyone()
        checkpoint_model(
            accelerator, model, tokenizer, opt.output_ckpt_dir, global_completed_steps
        )
        if opt.save_all_states:
            checkpoint_model_optimizer_scheduler(
                opt.output_ckpt_dir,
                model,
                global_completed_steps,
                lr_scheduler,
                accelerator,
            )
        # --
        # Check for convergence at the end of each epoch
        if opt.kd_loss is not None:
            threshold = 1e-27
            patience = opt.epochs
        else:
            threshold = 1e-4
            patience = 2

        if check_convergence(
            accelerator, losses, patience=patience, threshold=threshold
        ):
            accelerator.print(f"**** Model has converged at {epoch} epoch.")
            break


if __name__ == "__main__":
    opt = parse_option()
    train(opt)
