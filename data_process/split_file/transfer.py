import subprocess
import sys
import os
import tempfile

# 配置参数（需要根据你的实际情况修改）
LOCAL_USER = 'wuguocheng'  # 本地机器的用户名
LOCAL_HOST = '10.200.2.119'   # 本地机器的 IP 地址或域名
SSH_KEY = os.path.expanduser('~/.ssh/id_ed25519')  # REMOTE_HOST 上的 SSH 私钥路径

def main():
    # 检查命令行参数
    if len(sys.argv) != 2:
        print("用法: python script.py <txt_file>")
        sys.exit(1)
    
    txt_file = sys.argv[1]
    
    # 验证 txt 文件是否存在
    if not os.path.isfile(txt_file):
        print(f"错误: {txt_file} 不存在或不是文件")
        sys.exit(1)
    
    # 读取 txt 文件中的路径
    with open(txt_file, "r") as f:
        lines = f.readlines()
    
    path_list = [line.strip() for line in lines if line.strip()]
    
    # 创建临时文件并写入路径
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        for path in path_list:
            temp_file.write(path + '\n')
        temp_file_name = temp_file.name
    
    # 构造 rsync 命令
    rsync_cmd = [
        'rsync',           # rsync 命令
        '-arv',            # 归档模式，递归传输并保留文件属性
        '--progress',      # 显示传输进度
        '-z',              # 启用压缩减少网络传输量
        '-e', f'ssh -i {SSH_KEY}',  # 使用指定的 SSH 私钥
        '--files-from', temp_file_name,  # 从临时文件中读取文件列表
        f'{LOCAL_USER}@{LOCAL_HOST}:',  # 源地址（本地机器）
        '/home/bingxing2/ailab/group/ai4astro/Datasets/AS/origin_fits'  # 目标目录（REMOTE_HOST 当前目录，可修改）
    ]
    
    # 显示将要执行的命令
    print("执行的命令:", ' '.join(rsync_cmd))
    
    # 执行 rsync 命令
    result = subprocess.run(rsync_cmd)
    
    # 删除临时文件
    os.remove(temp_file_name)
    
    # 检查执行结果
    if result.returncode != 0:
        print(f"rsync 传输失败，返回码: {result.returncode}")
    else:
        print("文件传输成功完成")

if __name__ == '__main__':
    main()