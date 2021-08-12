# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/calculators/tensor/inference_calculator.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from mediapipe.framework import calculator_pb2 as mediapipe_dot_framework_dot_calculator__pb2
try:
  mediapipe_dot_framework_dot_calculator__options__pb2 = mediapipe_dot_framework_dot_calculator__pb2.mediapipe_dot_framework_dot_calculator__options__pb2
except AttributeError:
  mediapipe_dot_framework_dot_calculator__options__pb2 = mediapipe_dot_framework_dot_calculator__pb2.mediapipe.framework.calculator_options_pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='mediapipe/calculators/tensor/inference_calculator.proto',
  package='mediapipe',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n7mediapipe/calculators/tensor/inference_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\"\xc6\x08\n\x1aInferenceCalculatorOptions\x12\x12\n\nmodel_path\x18\x01 \x01(\t\x12\x1a\n\x07use_gpu\x18\x02 \x01(\x08:\x05\x66\x61lseB\x02\x18\x01\x12\x1c\n\tuse_nnapi\x18\x03 \x01(\x08:\x05\x66\x61lseB\x02\x18\x01\x12\x1a\n\x0e\x63pu_num_thread\x18\x04 \x01(\x05:\x02-1\x12@\n\x08\x64\x65legate\x18\x05 \x01(\x0b\x32..mediapipe.InferenceCalculatorOptions.Delegate\x1a\xa5\x06\n\x08\x44\x65legate\x12G\n\x06tflite\x18\x01 \x01(\x0b\x32\x35.mediapipe.InferenceCalculatorOptions.Delegate.TfLiteH\x00\x12\x41\n\x03gpu\x18\x02 \x01(\x0b\x32\x32.mediapipe.InferenceCalculatorOptions.Delegate.GpuH\x00\x12\x45\n\x05nnapi\x18\x03 \x01(\x0b\x32\x34.mediapipe.InferenceCalculatorOptions.Delegate.NnapiH\x00\x12I\n\x07xnnpack\x18\x04 \x01(\x0b\x32\x36.mediapipe.InferenceCalculatorOptions.Delegate.XnnpackH\x00\x1a\x08\n\x06TfLite\x1a\x8f\x03\n\x03Gpu\x12#\n\x14use_advanced_gpu_api\x18\x01 \x01(\x08:\x05\x66\x61lse\x12H\n\x03\x61pi\x18\x04 \x01(\x0e\x32\x36.mediapipe.InferenceCalculatorOptions.Delegate.Gpu.Api:\x03\x41NY\x12\"\n\x14\x61llow_precision_loss\x18\x03 \x01(\x08:\x04true\x12\x1a\n\x12\x63\x61\x63hed_kernel_path\x18\x02 \x01(\t\x12\x61\n\x05usage\x18\x05 \x01(\x0e\x32\x41.mediapipe.InferenceCalculatorOptions.Delegate.Gpu.InferenceUsage:\x0fSUSTAINED_SPEED\"&\n\x03\x41pi\x12\x07\n\x03\x41NY\x10\x00\x12\n\n\x06OPENGL\x10\x01\x12\n\n\x06OPENCL\x10\x02\"N\n\x0eInferenceUsage\x12\x0f\n\x0bUNSPECIFIED\x10\x00\x12\x16\n\x12\x46\x41ST_SINGLE_ANSWER\x10\x01\x12\x13\n\x0fSUSTAINED_SPEED\x10\x02\x1a/\n\x05Nnapi\x12\x11\n\tcache_dir\x18\x01 \x01(\t\x12\x13\n\x0bmodel_token\x18\x02 \x01(\t\x1a\"\n\x07Xnnpack\x12\x17\n\x0bnum_threads\x18\x01 \x01(\x05:\x02-1B\n\n\x08\x64\x65legate2T\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\xf7\xd3\xcb\xa0\x01 \x01(\x0b\x32%.mediapipe.InferenceCalculatorOptions')
  ,
  dependencies=[mediapipe_dot_framework_dot_calculator__pb2.DESCRIPTOR,])



_INFERENCECALCULATOROPTIONS_DELEGATE_GPU_API = _descriptor.EnumDescriptor(
  name='Api',
  full_name='mediapipe.InferenceCalculatorOptions.Delegate.Gpu.Api',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='ANY', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OPENGL', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OPENCL', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=902,
  serialized_end=940,
)
_sym_db.RegisterEnumDescriptor(_INFERENCECALCULATOROPTIONS_DELEGATE_GPU_API)

_INFERENCECALCULATOROPTIONS_DELEGATE_GPU_INFERENCEUSAGE = _descriptor.EnumDescriptor(
  name='InferenceUsage',
  full_name='mediapipe.InferenceCalculatorOptions.Delegate.Gpu.InferenceUsage',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNSPECIFIED', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FAST_SINGLE_ANSWER', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SUSTAINED_SPEED', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=942,
  serialized_end=1020,
)
_sym_db.RegisterEnumDescriptor(_INFERENCECALCULATOROPTIONS_DELEGATE_GPU_INFERENCEUSAGE)


_INFERENCECALCULATOROPTIONS_DELEGATE_TFLITE = _descriptor.Descriptor(
  name='TfLite',
  full_name='mediapipe.InferenceCalculatorOptions.Delegate.TfLite',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=610,
  serialized_end=618,
)

_INFERENCECALCULATOROPTIONS_DELEGATE_GPU = _descriptor.Descriptor(
  name='Gpu',
  full_name='mediapipe.InferenceCalculatorOptions.Delegate.Gpu',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='use_advanced_gpu_api', full_name='mediapipe.InferenceCalculatorOptions.Delegate.Gpu.use_advanced_gpu_api', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='api', full_name='mediapipe.InferenceCalculatorOptions.Delegate.Gpu.api', index=1,
      number=4, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='allow_precision_loss', full_name='mediapipe.InferenceCalculatorOptions.Delegate.Gpu.allow_precision_loss', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='cached_kernel_path', full_name='mediapipe.InferenceCalculatorOptions.Delegate.Gpu.cached_kernel_path', index=3,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='usage', full_name='mediapipe.InferenceCalculatorOptions.Delegate.Gpu.usage', index=4,
      number=5, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=2,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _INFERENCECALCULATOROPTIONS_DELEGATE_GPU_API,
    _INFERENCECALCULATOROPTIONS_DELEGATE_GPU_INFERENCEUSAGE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=621,
  serialized_end=1020,
)

_INFERENCECALCULATOROPTIONS_DELEGATE_NNAPI = _descriptor.Descriptor(
  name='Nnapi',
  full_name='mediapipe.InferenceCalculatorOptions.Delegate.Nnapi',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='cache_dir', full_name='mediapipe.InferenceCalculatorOptions.Delegate.Nnapi.cache_dir', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model_token', full_name='mediapipe.InferenceCalculatorOptions.Delegate.Nnapi.model_token', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1022,
  serialized_end=1069,
)

_INFERENCECALCULATOROPTIONS_DELEGATE_XNNPACK = _descriptor.Descriptor(
  name='Xnnpack',
  full_name='mediapipe.InferenceCalculatorOptions.Delegate.Xnnpack',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_threads', full_name='mediapipe.InferenceCalculatorOptions.Delegate.Xnnpack.num_threads', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=-1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1071,
  serialized_end=1105,
)

_INFERENCECALCULATOROPTIONS_DELEGATE = _descriptor.Descriptor(
  name='Delegate',
  full_name='mediapipe.InferenceCalculatorOptions.Delegate',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='tflite', full_name='mediapipe.InferenceCalculatorOptions.Delegate.tflite', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='gpu', full_name='mediapipe.InferenceCalculatorOptions.Delegate.gpu', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nnapi', full_name='mediapipe.InferenceCalculatorOptions.Delegate.nnapi', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='xnnpack', full_name='mediapipe.InferenceCalculatorOptions.Delegate.xnnpack', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_INFERENCECALCULATOROPTIONS_DELEGATE_TFLITE, _INFERENCECALCULATOROPTIONS_DELEGATE_GPU, _INFERENCECALCULATOROPTIONS_DELEGATE_NNAPI, _INFERENCECALCULATOROPTIONS_DELEGATE_XNNPACK, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='delegate', full_name='mediapipe.InferenceCalculatorOptions.Delegate.delegate',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=312,
  serialized_end=1117,
)

_INFERENCECALCULATOROPTIONS = _descriptor.Descriptor(
  name='InferenceCalculatorOptions',
  full_name='mediapipe.InferenceCalculatorOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='model_path', full_name='mediapipe.InferenceCalculatorOptions.model_path', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='use_gpu', full_name='mediapipe.InferenceCalculatorOptions.use_gpu', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\030\001'), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='use_nnapi', full_name='mediapipe.InferenceCalculatorOptions.use_nnapi', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\030\001'), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='cpu_num_thread', full_name='mediapipe.InferenceCalculatorOptions.cpu_num_thread', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=-1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='delegate', full_name='mediapipe.InferenceCalculatorOptions.delegate', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
    _descriptor.FieldDescriptor(
      name='ext', full_name='mediapipe.InferenceCalculatorOptions.ext', index=0,
      number=336783863, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=True, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  nested_types=[_INFERENCECALCULATOROPTIONS_DELEGATE, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=109,
  serialized_end=1203,
)

_INFERENCECALCULATOROPTIONS_DELEGATE_TFLITE.containing_type = _INFERENCECALCULATOROPTIONS_DELEGATE
_INFERENCECALCULATOROPTIONS_DELEGATE_GPU.fields_by_name['api'].enum_type = _INFERENCECALCULATOROPTIONS_DELEGATE_GPU_API
_INFERENCECALCULATOROPTIONS_DELEGATE_GPU.fields_by_name['usage'].enum_type = _INFERENCECALCULATOROPTIONS_DELEGATE_GPU_INFERENCEUSAGE
_INFERENCECALCULATOROPTIONS_DELEGATE_GPU.containing_type = _INFERENCECALCULATOROPTIONS_DELEGATE
_INFERENCECALCULATOROPTIONS_DELEGATE_GPU_API.containing_type = _INFERENCECALCULATOROPTIONS_DELEGATE_GPU
_INFERENCECALCULATOROPTIONS_DELEGATE_GPU_INFERENCEUSAGE.containing_type = _INFERENCECALCULATOROPTIONS_DELEGATE_GPU
_INFERENCECALCULATOROPTIONS_DELEGATE_NNAPI.containing_type = _INFERENCECALCULATOROPTIONS_DELEGATE
_INFERENCECALCULATOROPTIONS_DELEGATE_XNNPACK.containing_type = _INFERENCECALCULATOROPTIONS_DELEGATE
_INFERENCECALCULATOROPTIONS_DELEGATE.fields_by_name['tflite'].message_type = _INFERENCECALCULATOROPTIONS_DELEGATE_TFLITE
_INFERENCECALCULATOROPTIONS_DELEGATE.fields_by_name['gpu'].message_type = _INFERENCECALCULATOROPTIONS_DELEGATE_GPU
_INFERENCECALCULATOROPTIONS_DELEGATE.fields_by_name['nnapi'].message_type = _INFERENCECALCULATOROPTIONS_DELEGATE_NNAPI
_INFERENCECALCULATOROPTIONS_DELEGATE.fields_by_name['xnnpack'].message_type = _INFERENCECALCULATOROPTIONS_DELEGATE_XNNPACK
_INFERENCECALCULATOROPTIONS_DELEGATE.containing_type = _INFERENCECALCULATOROPTIONS
_INFERENCECALCULATOROPTIONS_DELEGATE.oneofs_by_name['delegate'].fields.append(
  _INFERENCECALCULATOROPTIONS_DELEGATE.fields_by_name['tflite'])
_INFERENCECALCULATOROPTIONS_DELEGATE.fields_by_name['tflite'].containing_oneof = _INFERENCECALCULATOROPTIONS_DELEGATE.oneofs_by_name['delegate']
_INFERENCECALCULATOROPTIONS_DELEGATE.oneofs_by_name['delegate'].fields.append(
  _INFERENCECALCULATOROPTIONS_DELEGATE.fields_by_name['gpu'])
_INFERENCECALCULATOROPTIONS_DELEGATE.fields_by_name['gpu'].containing_oneof = _INFERENCECALCULATOROPTIONS_DELEGATE.oneofs_by_name['delegate']
_INFERENCECALCULATOROPTIONS_DELEGATE.oneofs_by_name['delegate'].fields.append(
  _INFERENCECALCULATOROPTIONS_DELEGATE.fields_by_name['nnapi'])
_INFERENCECALCULATOROPTIONS_DELEGATE.fields_by_name['nnapi'].containing_oneof = _INFERENCECALCULATOROPTIONS_DELEGATE.oneofs_by_name['delegate']
_INFERENCECALCULATOROPTIONS_DELEGATE.oneofs_by_name['delegate'].fields.append(
  _INFERENCECALCULATOROPTIONS_DELEGATE.fields_by_name['xnnpack'])
_INFERENCECALCULATOROPTIONS_DELEGATE.fields_by_name['xnnpack'].containing_oneof = _INFERENCECALCULATOROPTIONS_DELEGATE.oneofs_by_name['delegate']
_INFERENCECALCULATOROPTIONS.fields_by_name['delegate'].message_type = _INFERENCECALCULATOROPTIONS_DELEGATE
DESCRIPTOR.message_types_by_name['InferenceCalculatorOptions'] = _INFERENCECALCULATOROPTIONS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

InferenceCalculatorOptions = _reflection.GeneratedProtocolMessageType('InferenceCalculatorOptions', (_message.Message,), dict(

  Delegate = _reflection.GeneratedProtocolMessageType('Delegate', (_message.Message,), dict(

    TfLite = _reflection.GeneratedProtocolMessageType('TfLite', (_message.Message,), dict(
      DESCRIPTOR = _INFERENCECALCULATOROPTIONS_DELEGATE_TFLITE,
      __module__ = 'mediapipe.calculators.tensor.inference_calculator_pb2'
      # @@protoc_insertion_point(class_scope:mediapipe.InferenceCalculatorOptions.Delegate.TfLite)
      ))
    ,

    Gpu = _reflection.GeneratedProtocolMessageType('Gpu', (_message.Message,), dict(
      DESCRIPTOR = _INFERENCECALCULATOROPTIONS_DELEGATE_GPU,
      __module__ = 'mediapipe.calculators.tensor.inference_calculator_pb2'
      # @@protoc_insertion_point(class_scope:mediapipe.InferenceCalculatorOptions.Delegate.Gpu)
      ))
    ,

    Nnapi = _reflection.GeneratedProtocolMessageType('Nnapi', (_message.Message,), dict(
      DESCRIPTOR = _INFERENCECALCULATOROPTIONS_DELEGATE_NNAPI,
      __module__ = 'mediapipe.calculators.tensor.inference_calculator_pb2'
      # @@protoc_insertion_point(class_scope:mediapipe.InferenceCalculatorOptions.Delegate.Nnapi)
      ))
    ,

    Xnnpack = _reflection.GeneratedProtocolMessageType('Xnnpack', (_message.Message,), dict(
      DESCRIPTOR = _INFERENCECALCULATOROPTIONS_DELEGATE_XNNPACK,
      __module__ = 'mediapipe.calculators.tensor.inference_calculator_pb2'
      # @@protoc_insertion_point(class_scope:mediapipe.InferenceCalculatorOptions.Delegate.Xnnpack)
      ))
    ,
    DESCRIPTOR = _INFERENCECALCULATOROPTIONS_DELEGATE,
    __module__ = 'mediapipe.calculators.tensor.inference_calculator_pb2'
    # @@protoc_insertion_point(class_scope:mediapipe.InferenceCalculatorOptions.Delegate)
    ))
  ,
  DESCRIPTOR = _INFERENCECALCULATOROPTIONS,
  __module__ = 'mediapipe.calculators.tensor.inference_calculator_pb2'
  # @@protoc_insertion_point(class_scope:mediapipe.InferenceCalculatorOptions)
  ))
_sym_db.RegisterMessage(InferenceCalculatorOptions)
_sym_db.RegisterMessage(InferenceCalculatorOptions.Delegate)
_sym_db.RegisterMessage(InferenceCalculatorOptions.Delegate.TfLite)
_sym_db.RegisterMessage(InferenceCalculatorOptions.Delegate.Gpu)
_sym_db.RegisterMessage(InferenceCalculatorOptions.Delegate.Nnapi)
_sym_db.RegisterMessage(InferenceCalculatorOptions.Delegate.Xnnpack)

_INFERENCECALCULATOROPTIONS.extensions_by_name['ext'].message_type = _INFERENCECALCULATOROPTIONS
mediapipe_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_INFERENCECALCULATOROPTIONS.extensions_by_name['ext'])

_INFERENCECALCULATOROPTIONS.fields_by_name['use_gpu']._options = None
_INFERENCECALCULATOROPTIONS.fields_by_name['use_nnapi']._options = None
# @@protoc_insertion_point(module_scope)
