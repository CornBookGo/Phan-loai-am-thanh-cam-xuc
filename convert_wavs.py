

"""
Lệnh sử dụng để chuyển đổi các mẫu âm thanh sang âm thanh thích hợp cho trích xuất đặc trưng
"""

import os

def convert_audio(audio_path, target_path, remove=False):
    """Chuc nang nay dat am thanh tu 'audio_path' thanh:
        - 16000Hz Sampling rate
        - 1 kenh am thanh ( mono )
            Thong so:
                audio_path (str): duong dan tap am thanh wav muon chuyen doi
                target_path (str): duong dan de luu tep am thanh moi da chuyen doi 
                remove (bool): co nen xoa file cu sau khi chuyen doi khong
        Luu y chuc nang can phai cai dat ffmpeg trong he thong Path tren may tinh."""

    v = os.system(f"ffmpeg -i {audio_path} -ac 1 -ar 16000 {target_path}")
    # os.system(f"ffmpeg -i {audio_path} -ac 1 {target_path}")
    if remove:
        os.remove(audio_path)
    return v


def convert_audios(path, target_path, remove=False):
    """Chuyen doi duong dan cua tep wav sang:
        - 16000Hz Sampling rate
        - one audio channel ( mono )
        va dat chung vao mot thu muc moi co ten 'target_path'
            Thong so:
                audio_path (str): duong dan cua am thanh wav muon chuyen doi
                target_path (str): duong dan de luu tep am thanh moi da chuyen doi 
                remove (bool): co nen xoa file cu sau khi chuyen doi khong
        Luu y chuc nang can phai cai dat ffmpeg trong he thong Path tren may tinh."""

    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            dirname = os.path.join(dirpath, dirname)
            target_dir = dirname.replace(path, target_path)
            if not os.path.isdir(target_dir):
                os.mkdir(target_dir)

    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            file = os.path.join(dirpath, filename)
            if file.endswith(".wav"):
                # it is a wav file
                target_file = file.replace(path, target_path)
                convert_audio(file, target_file, remove=remove)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="""Chuyen doi (nen) tap tin wav thanh am thanh mono va 16000Hz (mono). Tien ich nay giup nen cac tap tin wav de dao tao va kiem tra""")
    parser.add_argument("audio_path", help="Thu muc chua file wav muon chuyen doi")
    parser.add_argument("target_path", help="Thu muc luu file wav moi")
    parser.add_argument("-r", "--remove", type=bool, help="Co nen xoa file wav cu sau khi chuyen doi hay khong", default=False)

    args = parser.parse_args()
    audio_path = args.audio_path
    target_path = args.target_path

    if os.path.isdir(audio_path):
        if not os.path.isdir(target_path):
            os.makedirs(target_path)
            convert_audios(audio_path, target_path, remove=args.remove)
    elif os.path.isfile(audio_path) and audio_path.endswith(".wav"):
        if not target_path.endswith(".wav"):
            target_path += ".wav"
        convert_audio(audio_path, target_path, remove=args.remove)
    else:
        raise TypeError("Tep audio_path chi dinh khong phu hop cho thao tac nay")
