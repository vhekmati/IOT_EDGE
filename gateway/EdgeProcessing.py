#!/usr/bin/env python3

import base64
import csv
import ctypes
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tflite_runtime.interpreter import Interpreter


DEFAULT_KEY_HEX = (
    "00112233445566778899aabbccddeeff"
    "00112233445566778899aabbccddeeff"
)


class EdgeProcessing:
    def __init__(self, name: str):
        self.name = name

    def process(self, message):
        raise NotImplementedError


class DataAnalysisProcessor(EdgeProcessing):
    def __init__(self, name: str, backend: str):
        super().__init__(name)
        self.backend = backend

    def process(self, message):
        if self.backend == "python":
            data = message["data"]
            feature_count = len(data[0])
            running_mean = [0.0] * feature_count
            cma = []

            for row_index, row in enumerate(data):
                count = row_index + 1
                for feature_index in range(feature_count):
                    running_mean[feature_index] += (
                        row[feature_index] - running_mean[feature_index]
                    ) / count
                cma.append(running_mean.copy())

            message["data"] = cma
            return message

        if self.backend == "numpy":
            values = np.asarray(message["data"], dtype=np.float32)
            divisors = np.arange(
                1,
                values.shape[0] + 1,
                dtype=np.float32,
            ).reshape(-1, 1)
            message["data"] = np.cumsum(values, axis=0) / divisors
            return message

        raise ValueError(f"Unknown analytics backend: {self.backend}")


class AIProcessingProcessor(EdgeProcessing):
    def __init__(self, name: str, backend: str, runtime_config: dict):
        super().__init__(name)
        self.backend = backend
        self.runtime_config = runtime_config
        self._local = threading.local()

        model_dir = Path(runtime_config["model_artifacts_dir"])
        scaler = np.load(model_dir / "norm_params_medium.npz")

        self.window = int(scaler["WINDOW"])
        self.dmin = scaler["dmin"].astype(np.float32)
        self.scale = scaler["scale"].astype(np.float32)
        self.scale[self.scale < 1e-6] = 1.0

        self.class_names = {
            0: "healthy",
            1: "warning",
            2: "failure",
        }

        if backend == "cpu":
            self.model_path = str(model_dir / "cmapss_medium_cnn_int8.tflite")
        elif backend == "edgetpu":
            self.model_path = str(
                model_dir / "cmapss_medium_cnn_int8_edgetpu.tflite"
            )
        else:
            raise ValueError(f"Unknown AI backend: {backend}")

    def _build_interpreter(self):
        if self.backend == "cpu":
            interpreter = Interpreter(model_path=self.model_path)
        else:
            libs_dir = self.runtime_config["libs_dir"]
            libusb_path = os.path.join(libs_dir, "libusb-1.0.so.0")
            edgetpu_path = os.path.join(libs_dir, "libedgetpu.so.1")

            ctypes.CDLL(libusb_path, ctypes.RTLD_GLOBAL)
            ctypes.CDLL(edgetpu_path, ctypes.RTLD_GLOBAL)

            from tflite_runtime.interpreter import load_delegate

            delegate = load_delegate(edgetpu_path)
            interpreter = Interpreter(
                model_path=self.model_path,
                experimental_delegates=[delegate],
            )

        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        self._local.interpreter = interpreter
        self._local.input_details = input_details
        self._local.output_details = output_details
        self._local.input_scale = float(
            input_details["quantization_parameters"]["scales"][0]
        )
        self._local.input_zero_point = int(
            input_details["quantization_parameters"]["zero_points"][0]
        )
        self._local.output_scale = float(
            output_details["quantization_parameters"]["scales"][0]
        )
        self._local.output_zero_point = int(
            output_details["quantization_parameters"]["zero_points"][0]
        )

        dummy = np.zeros(input_details["shape"], dtype=np.int8)
        interpreter.set_tensor(input_details["index"], dummy)
        for _ in range(3):
            interpreter.invoke()

    def _ensure_initialized(self):
        if not hasattr(self._local, "interpreter"):
            self._build_interpreter()

    @staticmethod
    def _softmax(logits):
        shifted = logits - np.max(logits)
        exponentials = np.exp(shifted)
        return exponentials / np.sum(exponentials)

    def process(self, message):
        self._ensure_initialized()

        values = np.asarray(message["data"], dtype=np.float32)
        data = values[-self.window :, :]

        normalized = (data - self.dmin) / self.scale
        normalized = np.clip(normalized, 0.0, 1.0)
        normalized = normalized.reshape(1, self.window, 7, 1).astype(np.float32)

        quantized = np.round(
            normalized / self._local.input_scale
            + self._local.input_zero_point
        )
        quantized = np.clip(quantized, -128, 127).astype(np.int8)

        interpreter = self._local.interpreter
        interpreter.set_tensor(
            self._local.input_details["index"],
            quantized,
        )
        interpreter.invoke()

        raw_output = interpreter.get_tensor(
            self._local.output_details["index"]
        )
        logits = (
            raw_output.astype(np.float32)
            - self._local.output_zero_point
        ) * self._local.output_scale
        logits = logits[0]

        probabilities = self._softmax(logits)
        class_id = int(np.argmax(probabilities))

        return {
            "experiment_id": message.get("experiment_id", ""),
            "seq_id": message.get("seq_id", ""),
            "class_id": class_id,
            "class_name": self.class_names[class_id],
            "probabilities": {
                "healthy": round(float(probabilities[0]), 4),
                "warning": round(float(probabilities[1]), 4),
                "failure": round(float(probabilities[2]), 4),
            },
            "logits": [
                round(float(value), 4)
                for value in logits.tolist()
            ],
        }


class _PyCryptodomeBackend:
    def __init__(self, key: bytes):
        from Crypto.Cipher import AES
        from Crypto.Util.Padding import pad, unpad

        self.AES = AES
        self.pad = pad
        self.unpad = unpad
        self.key = key

    def encrypt(self, plaintext: bytes, iv: bytes) -> bytes:
        cipher = self.AES.new(self.key, self.AES.MODE_CBC, iv=iv)
        return cipher.encrypt(self.pad(plaintext, self.AES.block_size))

    def decrypt(self, ciphertext: bytes, iv: bytes) -> bytes:
        cipher = self.AES.new(self.key, self.AES.MODE_CBC, iv=iv)
        padded = cipher.decrypt(ciphertext)
        return self.unpad(padded, self.AES.block_size)

    def close(self):
        return None


class _OpenSSLBackend:
    def __init__(self, key: bytes):
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import padding
        from cryptography.hazmat.primitives.ciphers import (
            Cipher,
            algorithms,
            modes,
        )

        self.default_backend = default_backend
        self.padding = padding
        self.Cipher = Cipher
        self.algorithms = algorithms
        self.modes = modes
        self.key = key

    def encrypt(self, plaintext: bytes, iv: bytes) -> bytes:
        padder = self.padding.PKCS7(128).padder()
        padded = padder.update(plaintext) + padder.finalize()
        cipher = self.Cipher(
            self.algorithms.AES(self.key),
            self.modes.CBC(iv),
            backend=self.default_backend(),
        )
        encryptor = cipher.encryptor()
        return encryptor.update(padded) + encryptor.finalize()

    def decrypt(self, ciphertext: bytes, iv: bytes) -> bytes:
        cipher = self.Cipher(
            self.algorithms.AES(self.key),
            self.modes.CBC(iv),
            backend=self.default_backend(),
        )
        decryptor = cipher.decryptor()
        padded = decryptor.update(ciphertext) + decryptor.finalize()
        unpadder = self.padding.PKCS7(128).unpadder()
        return unpadder.update(padded) + unpadder.finalize()

    def close(self):
        return None


@dataclass
class _TitanSSLConfig:
    key_hex: str
    openssl_dir: Path
    engine_name: str
    engine_dir: Path
    libcrypto_path: Path


class _TitanSSLBackend:
    EVP_MAX_BLOCK_LENGTH = 32

    def __init__(self, config: _TitanSSLConfig):
        self.config = config
        self.key = bytes.fromhex(config.key_hex)

        os.environ["OPENSSL_ENGINES"] = str(config.engine_dir)
        self.libcrypto = ctypes.CDLL(
            str(config.libcrypto_path),
            mode=ctypes.RTLD_GLOBAL,
        )
        self._configure_openssl()

        # In OpenSSL 1.1.1, ENGINE_load_dynamic() is a C macro,
        # not an exported shared-library symbol. Its actual operation is:
        # OPENSSL_init_crypto(OPENSSL_INIT_ENGINE_DYNAMIC, NULL).
        OPENSSL_INIT_ENGINE_DYNAMIC = 0x00000400
        if self.libcrypto.OPENSSL_init_crypto(
            OPENSSL_INIT_ENGINE_DYNAMIC,
            None,
        ) != 1:
            raise RuntimeError(
                "OpenSSL dynamic ENGINE initialization failed"
            )

        self.engine = self.libcrypto.ENGINE_by_id(
            config.engine_name.encode("utf-8")
        )
        if not self.engine:
            raise RuntimeError("TitanSSL Engine could not be loaded")
        if self.libcrypto.ENGINE_init(self.engine) != 1:
            raise RuntimeError("TitanSSL Engine initialization failed")

        self.cipher = self.libcrypto.EVP_aes_256_cbc()

    def _configure_openssl(self):
        lib = self.libcrypto

        lib.OPENSSL_init_crypto.argtypes = [
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        lib.OPENSSL_init_crypto.restype = ctypes.c_int

        lib.ENGINE_by_id.argtypes = [ctypes.c_char_p]
        lib.ENGINE_by_id.restype = ctypes.c_void_p
        lib.ENGINE_init.argtypes = [ctypes.c_void_p]
        lib.ENGINE_init.restype = ctypes.c_int
        lib.ENGINE_finish.argtypes = [ctypes.c_void_p]
        lib.ENGINE_finish.restype = ctypes.c_int
        lib.ENGINE_free.argtypes = [ctypes.c_void_p]
        lib.ENGINE_free.restype = ctypes.c_int
        lib.EVP_aes_256_cbc.argtypes = []
        lib.EVP_aes_256_cbc.restype = ctypes.c_void_p
        lib.EVP_CIPHER_CTX_new.argtypes = []
        lib.EVP_CIPHER_CTX_new.restype = ctypes.c_void_p
        lib.EVP_CIPHER_CTX_free.argtypes = [ctypes.c_void_p]
        lib.EVP_CIPHER_CTX_free.restype = None

        lib.EVP_EncryptInit_ex.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.EVP_EncryptInit_ex.restype = ctypes.c_int
        lib.EVP_EncryptUpdate.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.EVP_EncryptUpdate.restype = ctypes.c_int
        lib.EVP_EncryptFinal_ex.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.EVP_EncryptFinal_ex.restype = ctypes.c_int

        lib.EVP_DecryptInit_ex.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.EVP_DecryptInit_ex.restype = ctypes.c_int
        lib.EVP_DecryptUpdate.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.EVP_DecryptUpdate.restype = ctypes.c_int
        lib.EVP_DecryptFinal_ex.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.EVP_DecryptFinal_ex.restype = ctypes.c_int

    def _crypt(self, data: bytes, iv: bytes, encrypt: bool) -> bytes:
        context = self.libcrypto.EVP_CIPHER_CTX_new()
        if not context:
            raise RuntimeError("EVP_CIPHER_CTX_new failed")

        input_buffer = ctypes.create_string_buffer(data, len(data))
        key_buffer = ctypes.create_string_buffer(self.key, len(self.key))
        iv_buffer = ctypes.create_string_buffer(iv, len(iv))
        output_buffer = ctypes.create_string_buffer(
            len(data) + self.EVP_MAX_BLOCK_LENGTH
        )
        update_length = ctypes.c_int()
        final_length = ctypes.c_int()

        if encrypt:
            init = self.libcrypto.EVP_EncryptInit_ex
            update = self.libcrypto.EVP_EncryptUpdate
            final = self.libcrypto.EVP_EncryptFinal_ex
        else:
            init = self.libcrypto.EVP_DecryptInit_ex
            update = self.libcrypto.EVP_DecryptUpdate
            final = self.libcrypto.EVP_DecryptFinal_ex

        try:
            if init(
                context,
                self.cipher,
                self.engine,
                key_buffer,
                iv_buffer,
            ) != 1:
                raise RuntimeError("EVP initialization failed")

            if update(
                context,
                output_buffer,
                ctypes.byref(update_length),
                input_buffer,
                len(data),
            ) != 1:
                raise RuntimeError("EVP update failed")

            final_pointer = ctypes.byref(
                output_buffer,
                update_length.value,
            )
            if final(
                context,
                final_pointer,
                ctypes.byref(final_length),
            ) != 1:
                raise RuntimeError("EVP finalization failed")

            output_length = update_length.value + final_length.value
            return output_buffer.raw[:output_length]
        finally:
            self.libcrypto.EVP_CIPHER_CTX_free(context)

    def encrypt(self, plaintext: bytes, iv: bytes) -> bytes:
        return self._crypt(plaintext, iv, encrypt=True)

    def decrypt(self, ciphertext: bytes, iv: bytes) -> bytes:
        return self._crypt(ciphertext, iv, encrypt=False)

    def close(self):
        self.libcrypto.ENGINE_finish(self.engine)
        self.libcrypto.ENGINE_free(self.engine)


class CyberSecurityProcessor(EdgeProcessing):
    ALGORITHM = "AES-256-CBC"

    def __init__(self, name: str, backend: str, runtime_config: dict):
        super().__init__(name)
        self.backend_name = backend
        self.key = bytes.fromhex(
            runtime_config.get("key_hex", DEFAULT_KEY_HEX)
        )

        if backend == "pycryptodome":
            self.backend = _PyCryptodomeBackend(self.key)
        elif backend == "openssl":
            self.backend = _OpenSSLBackend(self.key)
        elif backend == "optee":
            optee = runtime_config["optee"]
            openssl_dir = Path(optee["openssl_dir"])
            config = _TitanSSLConfig(
                key_hex=runtime_config.get("key_hex", DEFAULT_KEY_HEX),
                openssl_dir=openssl_dir,
                engine_name=optee.get(
                    "engine_name",
                    "titanssl_agnostic",
                ),
                engine_dir=Path(
                    optee.get(
                        "engine_dir",
                        str(openssl_dir / "engines"),
                    )
                ),
                libcrypto_path=Path(
                    optee.get(
                        "libcrypto_path",
                        str(openssl_dir / "lib" / "libcrypto.so.1.1"),
                    )
                ),
            )
            self.backend = _TitanSSLBackend(config)
        else:
            raise ValueError(f"Unknown crypto backend: {backend}")

    def encrypt_bytes(self, plaintext: bytes, iv: bytes) -> bytes:
        return self.backend.encrypt(plaintext, iv)

    def decrypt_bytes(self, ciphertext: bytes, iv: bytes) -> bytes:
        return self.backend.decrypt(ciphertext, iv)

    def close(self):
        self.backend.close()


class DecryptProcessor(CyberSecurityProcessor):
    def __init__(self, name: str, cybersecurity: CyberSecurityProcessor):
        EdgeProcessing.__init__(self, name)
        self.cybersecurity = cybersecurity

    def process(self, message):
        ciphertext, iv = message
        return self.cybersecurity.decrypt_bytes(ciphertext, iv)


class EncryptProcessor(CyberSecurityProcessor):
    def __init__(self, name: str, cybersecurity: CyberSecurityProcessor):
        EdgeProcessing.__init__(self, name)
        self.cybersecurity = cybersecurity

    def process(self, message):
        plaintext, iv = message
        return self.cybersecurity.encrypt_bytes(plaintext, iv)


class EdgeProcessingChain:
    MEASUREMENT_HEADER = [
        "experiment_id",
        "seq_id",
        "total_start_ns",
        "decrypt_start_ns",
        "decrypt_end_ns",
        "analytics_start_ns",
        "analytics_end_ns",
        "ai_start_ns",
        "ai_end_ns",
        "encrypt_start_ns",
        "encrypt_end_ns",
        "total_end_ns",
    ]

    def __init__(self, processors=None, device_id="unknown", config=None):
        del processors
        self.config = config or {}
        self.device_id = device_id
        self.mode = self.config["mode"]
        self.experiment_id = self.config["experiment_id"]
        self.output_dir = Path(self.config["output_directory"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        runtime_config = self.config["runtime"]

        self.cybersecurity = CyberSecurityProcessor(
            "CyberSecurity",
            self.config["crypto_backend"],
            runtime_config,
        )
        self.decrypt_processor = DecryptProcessor(
            "Decrypt",
            self.cybersecurity,
        )
        self.analytics_processor = DataAnalysisProcessor(
            "DataAnalysis",
            self.config["analytics_backend"],
        )
        self.ai_processor = AIProcessingProcessor(
            "AIProcessing",
            self.config["ai_backend"],
            runtime_config,
        )
        self.encrypt_processor = EncryptProcessor(
            "Encrypt",
            self.cybersecurity,
        )

        self.measurement_rows = []
        self.last_seq_id = ""
        self.last_experiment_id = ""

    @staticmethod
    def _decode_envelope(envelope_json: str):
        envelope = json.loads(envelope_json)
        iv = base64.b64decode(envelope["iv"])
        ciphertext = base64.b64decode(envelope["ct"])
        return iv, ciphertext

    @staticmethod
    def _encode_envelope(iv: bytes, ciphertext: bytes) -> str:
        envelope = {
            "v": 1,
            "alg": "AES-256-CBC",
            "iv": base64.b64encode(iv).decode("ascii"),
            "ct": base64.b64encode(ciphertext).decode("ascii"),
        }
        return json.dumps(envelope, separators=(",", ":"))

    def process(self, message: str) -> str:
        if self.mode == "measurement":
            return self._process_measurement(message)
        if self.mode == "deployment":
            return self._process_deployment(message)
        raise ValueError(f"Unknown mode: {self.mode}")

    def _process_measurement(self, message: str) -> str:
        total_start_ns = time.perf_counter_ns()

        iv, ciphertext = self._decode_envelope(message)

        decrypt_start_ns = time.perf_counter_ns()
        plaintext = self.decrypt_processor.process((ciphertext, iv))
        decrypt_end_ns = time.perf_counter_ns()

        payload = json.loads(plaintext.decode("utf-8"))
        self.last_seq_id = payload.get("seq_id", "")
        self.last_experiment_id = payload.get("experiment_id", "")
        if self.last_experiment_id != self.experiment_id:
            return None

        analytics_start_ns = time.perf_counter_ns()
        payload = self.analytics_processor.process(payload)
        analytics_end_ns = time.perf_counter_ns()

        ai_start_ns = time.perf_counter_ns()
        result = self.ai_processor.process(payload)
        ai_end_ns = time.perf_counter_ns()

        serialized_result = json.dumps(
            result,
            separators=(",", ":"),
        ).encode("utf-8")
        output_iv = os.urandom(16)

        encrypt_start_ns = time.perf_counter_ns()
        output_ciphertext = self.encrypt_processor.process(
            (serialized_result, output_iv)
        )
        encrypt_end_ns = time.perf_counter_ns()

        output = self._encode_envelope(output_iv, output_ciphertext)
        total_end_ns = time.perf_counter_ns()

        self.measurement_rows.append([
            self.experiment_id,
            self.last_seq_id,
            total_start_ns,
            decrypt_start_ns,
            decrypt_end_ns,
            analytics_start_ns,
            analytics_end_ns,
            ai_start_ns,
            ai_end_ns,
            encrypt_start_ns,
            encrypt_end_ns,
            total_end_ns,
        ])

        return output

    def _process_deployment(self, message: str) -> str:
        iv, ciphertext = self._decode_envelope(message)
        plaintext = self.decrypt_processor.process((ciphertext, iv))
        payload = json.loads(plaintext.decode("utf-8"))

        self.last_seq_id = payload.get("seq_id", "")
        self.last_experiment_id = payload.get("experiment_id", "")
        if self.last_experiment_id != self.experiment_id:
            return None

        payload = self.analytics_processor.process(payload)
        result = self.ai_processor.process(payload)

        serialized_result = json.dumps(
            result,
            separators=(",", ":"),
        ).encode("utf-8")
        output_iv = os.urandom(16)
        output_ciphertext = self.encrypt_processor.process(
            (serialized_result, output_iv)
        )
        return self._encode_envelope(output_iv, output_ciphertext)

    @staticmethod
    def _atomic_write_csv(path: Path, header, rows):
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as output_file:
            writer = csv.writer(output_file)
            writer.writerow(header)
            writer.writerows(rows)
        os.replace(temporary_path, path)

    def flush_latency(self):
        if self.mode != "measurement":
            return

        output_path = self.output_dir / "measurement_raw.csv"
        self._atomic_write_csv(
            output_path,
            self.MEASUREMENT_HEADER,
            self.measurement_rows,
        )
        self.measurement_rows.clear()

    def close(self):
        self.cybersecurity.close()
