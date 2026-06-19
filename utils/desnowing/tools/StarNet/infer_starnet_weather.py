import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf

import model


tf.disable_v2_behavior()

WINDOW_SIZE = 256
STRIDE = 64


def _first_image(input_dir: Path) -> Path:
	exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
	files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
	if not files:
		raise FileNotFoundError(f'No image found in {input_dir}')
	return files[0]


def _run_starnet(img_uint8: np.ndarray) -> np.ndarray:
	x = tf.placeholder(tf.float32, shape=[None, WINDOW_SIZE, WINDOW_SIZE, 3], name='X')
	y = tf.placeholder(tf.float32, shape=[None, WINDOW_SIZE, WINDOW_SIZE, 3], name='Y')
	_, _, outputs = model.model(x, y)

	init = tf.global_variables_initializer()
	saver = tf.train.Saver()

	image = img_uint8.astype(np.float32) / 255.0

	with tf.Session() as sess:
		sess.run(init)
		ckpt_prefix = str((Path(__file__).resolve().parent / 'model.ckpt'))
		saver.restore(sess, ckpt_prefix)

		offset = int((WINDOW_SIZE - STRIDE) / 2)
		h, w, _ = image.shape

		ith = int(h / STRIDE) + 1
		itw = int(w / STRIDE) + 1
		dh = ith * STRIDE - h
		dw = itw * STRIDE - w

		image = np.concatenate((image, image[(h - dh):, :, :]), axis=0)
		image = np.concatenate((image, image[:, (w - dw):, :]), axis=1)

		h, w, _ = image.shape
		image = np.concatenate((image, image[(h - offset):, :, :]), axis=0)
		image = np.concatenate((image[:offset, :, :], image), axis=0)
		image = np.concatenate((image, image[:, (w - offset):, :]), axis=1)
		image = np.concatenate((image[:, :offset, :], image), axis=1)

		output = np.copy(image)
		tmp = np.zeros((1, WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.float32)

		for i in range(ith):
			for j in range(itw):
				xx = STRIDE * i
				yy = STRIDE * j
				tmp[0] = image[xx: xx + WINDOW_SIZE, yy: yy + WINDOW_SIZE, :]
				result = sess.run(outputs, feed_dict={x: tmp})
				output[
					xx + offset: xx + STRIDE + offset,
					yy + offset: yy + STRIDE + offset,
					:
				] = result[0, offset: STRIDE + offset, offset: STRIDE + offset, :]

		output = np.clip(output, 0, 1)
		output = output[offset:-(offset + dh), offset:-(offset + dw), :]
		return (output * 255.0).astype(np.uint8)


def run(input_dir: Path, output_dir: Path) -> None:
	src = _first_image(input_dir)
	img = Image.open(src).convert('RGB')
	out_np = _run_starnet(np.array(img, dtype=np.uint8))

	output_dir.mkdir(parents=True, exist_ok=True)
	Image.fromarray(out_np).save(output_dir / 'output.png')


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_dir', type=str, required=True)
	parser.add_argument('--output_dir', type=str, required=True)
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()
	run(Path(args.input_dir), Path(args.output_dir))

