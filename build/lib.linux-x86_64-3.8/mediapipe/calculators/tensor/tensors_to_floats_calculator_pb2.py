# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/calculators/tensor/tensors_to_floats_calculator.proto

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
  name='mediapipe/calculators/tensor/tensors_to_floats_calculator.proto',
  package='mediapipe',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n?mediapipe/calculators/tensor/tensors_to_floats_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\"\xf5\x01\n TensorsToFloatsCalculatorOptions\x12P\n\nactivation\x18\x01 \x01(\x0e\x32\x36.mediapipe.TensorsToFloatsCalculatorOptions.Activation:\x04NONE\"#\n\nActivation\x12\x08\n\x04NONE\x10\x00\x12\x0b\n\x07SIGMOID\x10\x01\x32Z\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\xeb\xc2\xe5\xa3\x01 \x01(\x0b\x32+.mediapipe.TensorsToFloatsCalculatorOptions')
  ,
  dependencies=[mediapipe_dot_framework_dot_calculator__pb2.DESCRIPTOR,])



_TENSORSTOFLOATSCALCULATOROPTIONS_ACTIVATION = _descriptor.EnumDescriptor(
  name='Activation',
  full_name='mediapipe.TensorsToFloatsCalculatorOptions.Activation',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='NONE', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SIGMOID', index=1, number=1,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=235,
  serialized_end=270,
)
_sym_db.RegisterEnumDescriptor(_TENSORSTOFLOATSCALCULATOROPTIONS_ACTIVATION)


_TENSORSTOFLOATSCALCULATOROPTIONS = _descriptor.Descriptor(
  name='TensorsToFloatsCalculatorOptions',
  full_name='mediapipe.TensorsToFloatsCalculatorOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='activation', full_name='mediapipe.TensorsToFloatsCalculatorOptions.activation', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
    _descriptor.FieldDescriptor(
      name='ext', full_name='mediapipe.TensorsToFloatsCalculatorOptions.ext', index=0,
      number=343499115, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=True, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  nested_types=[],
  enum_types=[
    _TENSORSTOFLOATSCALCULATOROPTIONS_ACTIVATION,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=117,
  serialized_end=362,
)

_TENSORSTOFLOATSCALCULATOROPTIONS.fields_by_name['activation'].enum_type = _TENSORSTOFLOATSCALCULATOROPTIONS_ACTIVATION
_TENSORSTOFLOATSCALCULATOROPTIONS_ACTIVATION.containing_type = _TENSORSTOFLOATSCALCULATOROPTIONS
DESCRIPTOR.message_types_by_name['TensorsToFloatsCalculatorOptions'] = _TENSORSTOFLOATSCALCULATOROPTIONS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TensorsToFloatsCalculatorOptions = _reflection.GeneratedProtocolMessageType('TensorsToFloatsCalculatorOptions', (_message.Message,), dict(
  DESCRIPTOR = _TENSORSTOFLOATSCALCULATOROPTIONS,
  __module__ = 'mediapipe.calculators.tensor.tensors_to_floats_calculator_pb2'
  # @@protoc_insertion_point(class_scope:mediapipe.TensorsToFloatsCalculatorOptions)
  ))
_sym_db.RegisterMessage(TensorsToFloatsCalculatorOptions)

_TENSORSTOFLOATSCALCULATOROPTIONS.extensions_by_name['ext'].message_type = _TENSORSTOFLOATSCALCULATOROPTIONS
mediapipe_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_TENSORSTOFLOATSCALCULATOROPTIONS.extensions_by_name['ext'])

# @@protoc_insertion_point(module_scope)
