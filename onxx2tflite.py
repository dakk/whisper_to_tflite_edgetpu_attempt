from typing import Optional

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import shutil
import logging

import cv2
import numpy as np
import onnx
from onnx_tf.backend import prepare
import torch
import tensorflow as tf

import numpy as np
import random

def representative_dataset():
  for _ in range(100):
  #data = random.randint(0, 1)
  #yield [data]
    data = np.random.rand(32)*2
    yield [data.astype(np.float32)]
    
class Torch2TFLiteConverter:
    def __init__(
            self,
            onxx_model_path: str,
            tflite_model_save_path: str,
            sample_file_path: Optional[str] = None,
            target_shape: tuple = (224, 224, 3),
            seed: int = 10,
            normalize: bool = True
    ):
        self.onxx_model_path = onxx_model_path
        self.tflite_model_path = tflite_model_save_path
        self.sample_file_path = sample_file_path
        self.target_shape = target_shape
        self.seed = seed
        self.normalize = normalize

        self.tmpdir = '~/tmp/'
        self.__check_tmpdir()
        self.tf_model_path = os.path.join(self.tmpdir, 'tf_model')
        self.sample_data = self.load_sample_input(sample_file_path, target_shape, seed, normalize)

    def convert(self):
        self.onnx2tf()
        self.tf2tflite()

    def __check_tmpdir(self):
        try:
            if os.path.exists(self.tmpdir) and os.path.isdir(self.tmpdir):
                shutil.rmtree(self.tmpdir)
                logging.info(f'Old temp directory removed')
            os.makedirs(self.tmpdir, exist_ok=True)
            logging.info(f'Temp directory created at {self.tmpdir}')
        except Exception:
            logging.error('Can not create temporary directory, exiting!')
            sys.exit(-1)

    def load_tflite(self):

        interpret = tf.lite.Interpreter(self.tflite_model_path)
        interpret.allocate_tensors()
        logging.info(f'TFLite interpreter successfully loaded from, {self.tflite_model_path}')
        return interpret

    @staticmethod
    def load_sample_input(
            file_path: Optional[str] = None,
            target_shape: tuple = (224, 224, 3),
            seed: int = 10,
            normalize: bool = True
    ):
        if file_path is not None:
            if (len(target_shape) == 3 and target_shape[-1] == 1) or len(target_shape) == 2:
                imread_flags = cv2.IMREAD_GRAYSCALE
            elif len(target_shape) == 3 and target_shape[-1] == 3:
                imread_flags = cv2.IMREAD_COLOR
            else:
                imread_flags = cv2.IMREAD_ANYCOLOR + cv2.IMREAD_ANYDEPTH
            try:
                img = cv2.resize(
                    src=cv2.imread(file_path, imread_flags),
                    dsize=target_shape[:2],
                    interpolation=cv2.INTER_LINEAR
                )
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if normalize:
                    img = img * 1. / 255
                img = img.astype(np.float32)

                sample_data_np = np.transpose(img, (2, 0, 1))[np.newaxis, :, :, :]
                sample_data_torch = torch.from_numpy(sample_data_np)
                logging.info(f'Sample input successfully loaded from, {file_path}')

            except Exception:
                logging.error(f'Can not load sample input from, {file_path}')
                sys.exit(-1)

        else:
            logging.info(f'Sample input file path not specified, random data will be generated')
            np.random.seed(seed)
            data = np.random.random(target_shape).astype(np.float32)
            sample_data_np = np.transpose(data, (2, 0, 1))[np.newaxis, :, :, :]
            sample_data_torch = torch.from_numpy(sample_data_np)
            logging.info(f'Sample input randomly generated')

        return {'sample_data_np': sample_data_np, 'sample_data_torch': sample_data_torch}


    def onnx2tf(self) -> None:
        onnx_model = onnx.load(self.onxx_model_path)
        onnx.checker.check_model(onnx_model)
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(self.tf_model_path)

    def tf2tflite(self) -> None:
        converter = tf.lite.TFLiteConverter.from_saved_model(self.tf_model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.target_spec.supported_types = [tf.int8]
        converter.inference_input_type = tf.int8 # or tf.uint8 
        converter.inference_output_type = tf.int8 # or tf.uint8

        tflite_model = converter.convert()
        with open(self.tflite_model_path, 'wb') as f:
            f.write(tflite_model)

    def inference_torch(self) -> np.ndarray:
        y_pred = self.torch_model(self.sample_data['sample_data_torch'])
        return y_pred.detach().cpu().numpy()

    def inference_tflite(self, tflite_model) -> np.ndarray:
        input_details = tflite_model.get_input_details()
        output_details = tflite_model.get_output_details()
        tflite_model.set_tensor(input_details[0]['index'], self.sample_data['sample_data_np'])
        tflite_model.invoke()
        y_pred = tflite_model.get_tensor(output_details[0]['index'])
        return y_pred

    @staticmethod
    def calc_error(result_torch, result_tflite):
        mse = ((result_torch - result_tflite) ** 2).mean(axis=None)
        mae = np.abs(result_torch - result_tflite).mean(axis=None)
        logging.info(f'MSE (Mean-Square-Error): {mse}\tMAE (Mean-Absolute-Error): {mae}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--onxx-path', type=str, required=True)
    parser.add_argument('--tflite-path', type=str, required=True)
    parser.add_argument('--target-shape', type=tuple, nargs=3, default=(224, 224, 3))
    parser.add_argument('--sample-file', type=str)
    parser.add_argument('--seed', type=int, default=10)
    args = parser.parse_args()

    conv = Torch2TFLiteConverter(
        args.onxx_path,
        args.tflite_path,
        args.sample_file,
        args.target_shape,
        args.seed
    )
    conv.convert()
    sys.exit(0)
