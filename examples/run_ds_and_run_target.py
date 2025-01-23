"""
启动DS;并将训练脚本送入 DS


此文件不能正常运行 @250122

"""

import subprocess


deepspeed_cmd = [
    "deepspeed",
    "--num_gpus", "1",  # 使用的 GPU 数量
    "/home/fzm/Desktop/DeepSpeed/examples/dschats/training/step1_supervised_finetuning/main.py",  # 你的训练脚本
    # "--local_rank", "0",  # 显式指定本地 GPU 排序
    "--model_name_or_path", "/home/fzm/Desktop/DeepSpeed/models-files/facebook/opt-350m",  # 脚本参数
    "--gradient_accumulation_steps", "8",
    "--lora_dim","128",
    "--zero_stage","0",
    "--enable_tensorboard",
    "--tensorboard_path","/home/fzm/Desktop/DeepSpeedExamples/output",
    "--deepspeed",
    "--output_dir", "/home/fzm/Desktop/DeepSpeedExamples/output/training.log"
]

subprocess.run(deepspeed_cmd)



# from deepspeed.launcher.runner import main as deepspeed_runner

# # 模拟命令行参数
# args = [
#     "deepspeed",
#     "--num_gpus", "1",  # 使用的 GPU 数量
#     "/home/fzm/Desktop/DeepSpeed/examples/dschats/training/step1_supervised_finetuning/main.py",  # 你的训练脚本
#     "--",
#     "--model_name_or_path", "/home/fzm/Desktop/DeepSpeed/models-files/facebook/opt-350m",  # 脚本参数
#     "--gradient_accumulation_steps", "8",
#     "--lora_dim", "128",
#     "--zero_stage","0",
#     "--enable_tensorboard",
#     "--tensorboard_path","/home/fzm/Desktop/DeepSpeedExamples/output",
#     "--deepspeed",
#     "--output_dir", "/home/fzm/Desktop/DeepSpeedExamples/output/training.log"
# ]

# # 调用 DeepSpeed 主函数
# print("Args passed to deepspeed_runner:", args)
# res = deepspeed_runner(args)