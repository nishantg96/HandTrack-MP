# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/calculators/core/flow_limiter_calculator.proto

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
  name='mediapipe/calculators/core/flow_limiter_calculator.proto',
  package='mediapipe',
  syntax='proto2',
  serialized_options=_b('\242\002\tMediaPipe'),
  serialized_pb=_b('\n8mediapipe/calculators/core/flow_limiter_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\"\xcd\x01\n\x1c\x46lowLimiterCalculatorOptions\x12\x18\n\rmax_in_flight\x18\x01 \x01(\x05:\x01\x31\x12\x17\n\x0cmax_in_queue\x18\x02 \x01(\x05:\x01\x30\x12\"\n\x11in_flight_timeout\x18\x03 \x01(\x03:\x07\x31\x30\x30\x30\x30\x30\x30\x32V\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\xf8\xa0\xf4\x9b\x01 \x01(\x0b\x32\'.mediapipe.FlowLimiterCalculatorOptionsB\x0c\xa2\x02\tMediaPipe')
  ,
  dependencies=[mediapipe_dot_framework_dot_calculator__pb2.DESCRIPTOR,])




_FLOWLIMITERCALCULATOROPTIONS = _descriptor.Descriptor(
  name='FlowLimiterCalculatorOptions',
  full_name='mediapipe.FlowLimiterCalculatorOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='max_in_flight', full_name='mediapipe.FlowLimiterCalculatorOptions.max_in_flight', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_in_queue', full_name='mediapipe.FlowLimiterCalculatorOptions.max_in_queue', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='in_flight_timeout', full_name='mediapipe.FlowLimiterCalculatorOptions.in_flight_timeout', index=2,
      number=3, type=3, cpp_type=2, label=1,
      has_default_value=True, default_value=1000000,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
    _descriptor.FieldDescriptor(
      name='ext', full_name='mediapipe.FlowLimiterCalculatorOptions.ext', index=0,
      number=326963320, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=True, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
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
  serialized_start=110,
  serialized_end=315,
)

DESCRIPTOR.message_types_by_name['FlowLimiterCalculatorOptions'] = _FLOWLIMITERCALCULATOROPTIONS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

FlowLimiterCalculatorOptions = _reflection.GeneratedProtocolMessageType('FlowLimiterCalculatorOptions', (_message.Message,), dict(
  DESCRIPTOR = _FLOWLIMITERCALCULATOROPTIONS,
  __module__ = 'mediapipe.calculators.core.flow_limiter_calculator_pb2'
  # @@protoc_insertion_point(class_scope:mediapipe.FlowLimiterCalculatorOptions)
  ))
_sym_db.RegisterMessage(FlowLimiterCalculatorOptions)

_FLOWLIMITERCALCULATOROPTIONS.extensions_by_name['ext'].message_type = _FLOWLIMITERCALCULATOROPTIONS
mediapipe_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_FLOWLIMITERCALCULATOROPTIONS.extensions_by_name['ext'])

DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
