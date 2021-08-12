# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/modules/objectron/calculators/tensors_to_objects_calculator.proto

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
from mediapipe.modules.objectron.calculators import belief_decoder_config_pb2 as mediapipe_dot_modules_dot_objectron_dot_calculators_dot_belief__decoder__config__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='mediapipe/modules/objectron/calculators/tensors_to_objects_calculator.proto',
  package='mediapipe',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\nKmediapipe/modules/objectron/calculators/tensors_to_objects_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\x1a\x43mediapipe/modules/objectron/calculators/belief_decoder_config.proto\"\x88\x02\n!TensorsToObjectsCalculatorOptions\x12\x13\n\x0bnum_classes\x18\x01 \x01(\x05\x12\x15\n\rnum_keypoints\x18\x02 \x01(\x05\x12\"\n\x17num_values_per_keypoint\x18\x03 \x01(\x05:\x01\x32\x12\x36\n\x0e\x64\x65\x63oder_config\x18\x04 \x01(\x0b\x32\x1e.mediapipe.BeliefDecoderConfig2[\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\xd4\xea\xb7\x9f\x01 \x01(\x0b\x32,.mediapipe.TensorsToObjectsCalculatorOptions')
  ,
  dependencies=[mediapipe_dot_framework_dot_calculator__pb2.DESCRIPTOR,mediapipe_dot_modules_dot_objectron_dot_calculators_dot_belief__decoder__config__pb2.DESCRIPTOR,])




_TENSORSTOOBJECTSCALCULATOROPTIONS = _descriptor.Descriptor(
  name='TensorsToObjectsCalculatorOptions',
  full_name='mediapipe.TensorsToObjectsCalculatorOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_classes', full_name='mediapipe.TensorsToObjectsCalculatorOptions.num_classes', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_keypoints', full_name='mediapipe.TensorsToObjectsCalculatorOptions.num_keypoints', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_values_per_keypoint', full_name='mediapipe.TensorsToObjectsCalculatorOptions.num_values_per_keypoint', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=2,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='decoder_config', full_name='mediapipe.TensorsToObjectsCalculatorOptions.decoder_config', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
    _descriptor.FieldDescriptor(
      name='ext', full_name='mediapipe.TensorsToObjectsCalculatorOptions.ext', index=0,
      number=334361940, type=11, cpp_type=10, label=1,
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
  serialized_start=198,
  serialized_end=462,
)

_TENSORSTOOBJECTSCALCULATOROPTIONS.fields_by_name['decoder_config'].message_type = mediapipe_dot_modules_dot_objectron_dot_calculators_dot_belief__decoder__config__pb2._BELIEFDECODERCONFIG
DESCRIPTOR.message_types_by_name['TensorsToObjectsCalculatorOptions'] = _TENSORSTOOBJECTSCALCULATOROPTIONS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TensorsToObjectsCalculatorOptions = _reflection.GeneratedProtocolMessageType('TensorsToObjectsCalculatorOptions', (_message.Message,), dict(
  DESCRIPTOR = _TENSORSTOOBJECTSCALCULATOROPTIONS,
  __module__ = 'mediapipe.modules.objectron.calculators.tensors_to_objects_calculator_pb2'
  # @@protoc_insertion_point(class_scope:mediapipe.TensorsToObjectsCalculatorOptions)
  ))
_sym_db.RegisterMessage(TensorsToObjectsCalculatorOptions)

_TENSORSTOOBJECTSCALCULATOROPTIONS.extensions_by_name['ext'].message_type = _TENSORSTOOBJECTSCALCULATOROPTIONS
mediapipe_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_TENSORSTOOBJECTSCALCULATOROPTIONS.extensions_by_name['ext'])

# @@protoc_insertion_point(module_scope)
