import argparse
import os
import tensorflow as tf
from PIL import Image


parser = argparse.ArgumentParser(description='GIF generator from tensorboard event file')

parser.add_argument('--tb-file', help='Input tensorboard event file',
					default="runs/example/events.out.tfevents")
parser.add_argument('--out', help='Output path to generated GIF file',
					default="images/reconstruction_results.gif")
args = parser.parse_args()


im_str = tf.placeholder(tf.string)
im_tf = tf.image.decode_image(im_str)

sess = tf.InteractiveSession()

im_list = []

with sess.as_default():
	for e in tf.train.summary_iterator(args.tb_file):
		for v in e.summary.value:
			if v.tag.count("reconstructed/"):
				im = im_tf.eval({im_str: v.image.encoded_image_string})

				im = Image.fromarray(im)
				im_list.append(im)

im_list[0].save(args.out, save_all=True, append_images=im_list[1:], optimize=False, duration=60, loop=0)