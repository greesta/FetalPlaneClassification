import sys
import subprocess
import tensorflow as tf
from tensorflow.python.client import device_lib

# tf.py
# Singkat: cek apakah TensorFlow terpasang; jika terpasang tampilkan versi + GPU, jika belum beri instruksi.
# Cara pakai:
#   python tf.py          -> hanya cek
#   python tf.py --install -> coba pasang via pip lalu cek lagi


def print_install_instructions():
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    print("TensorFlow tidak ditemukan.")
    print("Instruksi singkat pemasangan:")
    print("  1) Pastikan Python >= 3.8 dan pip terbaru:")
    print("       python -m pip install --upgrade pip")
    print("  2) Pasang TensorFlow (CPU):")
    print("       python -m pip install tensorflow")
    print("  Jika ingin GPU, ikuti panduan resmi TensorFlow sesuai versi CUDA/cuDNN:")
    print("  https://www.tensorflow.org/install")

def try_install_tensorflow():
    print("Mencoba memasang TensorFlow lewat pip. Ini akan memanggil:")
    cmd = [sys.executable, "-m", "pip", "install", "tensorflow"]
    print("  " + " ".join(cmd))
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        print("Pemasangan gagal (pip mengembalikan error). Jalankan perintah di atas secara manual.")
        return False
    except Exception as e:
        print("Terjadi error saat mencoba pemasangan:", e)
        return False

def check_tensorflow():
    try:
        print("TensorFlow terpasang.")
        print("Versi:", tf.__version__)
        # Cek availability GPU (API berbeda-beda antar versi tapi ini umum)
        try:
            gpus = tf.config.list_physical_devices('GPU')
            print("Jumlah GPU yang terdeteksi:", len(gpus))
            if gpus:
                for i, g in enumerate(gpus):
                    print(f"  GPU {i}: {g}")
        except Exception:
            # Fallback untuk TF lama
            try:
                devs = device_lib.list_local_devices()
                gpus = [d for d in devs if d.device_type == 'GPU']
                print("Jumlah GPU yang terdeteksi (fallback):", len(gpus))
            except Exception:
                pass
        return True
    except ImportError:
        return False

def main():
    want_install = ("--install" in sys.argv)
    ok = check_tensorflow()
    if ok:
        return
    print_install_instructions()
    if want_install:
        if try_install_tensorflow():
            print("\nMencoba cek kembali setelah pemasangan...")
            if not check_tensorflow():
                print("Masih gagal mengimpor TensorFlow setelah pemasangan. Periksa pesan error di atas.")
        else:
            print("Pemasangan otomatis gagal. Coba jalankan perintah pip secara manual di Command Prompt/Terminal dengan hak akses yang sesuai.")

if __name__ == "__main__":
    main()