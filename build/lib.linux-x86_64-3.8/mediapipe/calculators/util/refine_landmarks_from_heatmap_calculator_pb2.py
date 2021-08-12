# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/calculators/util/refine_landmarks_from_heatmap_calculator.proto

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
  name='mediapipe/calculators/util/refine_landmarks_from_heatmap_calculator.proto',
  package='mediapipe',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\nImediapipe/calculators/util/refine_landmarks_from_heatmap_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\"\x95\x02\n+RefineLandmarksFromHeatmapCalculatorOptions\x12\x16\n\x0bkernel_size\x18\x01 \x01(\x05:\x01\x39\x12%\n\x18min_confidence_to_refine\x18\x02 \x01(\x02:\x03\x30.5\x12\x1e\n\x0frefine_presence\x18\x03 \x01(\x08:\x05\x66\x61lse\x12 \n\x11refine_visibility\x18\x04 \x01(\x08:\x05\x66\x61lse2e\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\xb5\xf5\xdf\xac\x01 \x01(\x0b\x32\x36.mediapipe.RefineLandmarksFromHeatmapCalculatorOptions')
  ,
  dependencies=[mediapipe_dot_framework_dot_calculator__pb2.DESCRIPTOR,])




_REFINELANDMARKSFROMHEATMAPCALCULATOROPTIONS = _descriptor.Descriptor(
  name='RefineLandmarksFromHeatmapCalculatorOptions',
  full_name='mediapipe.RefineLandmarksFromHeatmapCalculatorOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='kernel_size', full_name='mediapipe.RefineLandmarksFromHeatmapCalculatorOptions.kernel_size', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=9,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='min_confidence_to_refine', full_name='mediapipe.RefineLandmarksFromHeatmapCalculatorOptions.min_confidence_to_refine', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.5),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='refine_presence', full_name='mediapipe.RefineLandmarksFromHeatmapCalculatorOptions.refine_presence', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='refine_visibility', full_name='mediapipe.RefineLandmarksFromHeatmapCalculatorOptions.refine_visibility', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
    _descriptor.FieldDescriptor(
      name='ext', full_name='mediapipe.RefineLandmarksFromHeatmapCalculatorOptions.ext', index=0,
      number=362281653, type=11, cpp_type=10, label=1,
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
  serialized_start=127,
  serialized_end=404,
)

DESCRIPTOR.message_types_by_name['RefineLandmarksFromHeatmapCalculatorOptions'] = _REFINELANDMARKSFROMHEATMAPCALCULATOROPTIONS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

RefineLandmarksFromHeatmapCalculatorOptions = _reflection.GeneratedProtocolMessageType('RefineLandmarksFromHeatmapCalculatorOptions', (_message.Message,), dict(
  DESCRIPTOR = _REFINELANDMARKSFROMHEATMAPCALCULATOROPTIONS,
  __module__ = 'mediapipe.calculators.util.refine_landmarks_from_heatmap_calculator_pb2'
  # @@protoc_insertion_point(class_scope:mediapipe.RefineLandmarksFromHeatmapCalculatorOptions)
  ))
_sym_db.RegisterMessage(RefineLandmarksFromHeatmapCalculatorOptions)

_REFINELANDMARKSFROMHEATMAPCALCULATOROPTIONS.extensions_by_name['ext'].message_type = _REFINELANDMARKSFROMHEATMAPCALCULATOROPTIONS
mediapipe_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_REFINELANDMARKSFROMHEATMAPCALCULATOROPTIONS.extensions_by_name['ext'])

# @@protoc_insertion_point(module_scope)
