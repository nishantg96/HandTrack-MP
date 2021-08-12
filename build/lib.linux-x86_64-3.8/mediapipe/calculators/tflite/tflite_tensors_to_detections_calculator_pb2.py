# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto

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
  name='mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto',
  package='mediapipe',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\nJmediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\"\xf9\x04\n*TfLiteTensorsToDetectionsCalculatorOptions\x12\x13\n\x0bnum_classes\x18\x01 \x02(\x05\x12\x11\n\tnum_boxes\x18\x02 \x02(\x05\x12\x12\n\nnum_coords\x18\x03 \x02(\x05\x12\x1d\n\x15keypoint_coord_offset\x18\t \x01(\x05\x12\x18\n\rnum_keypoints\x18\n \x01(\x05:\x01\x30\x12\"\n\x17num_values_per_keypoint\x18\x0b \x01(\x05:\x01\x32\x12\x1b\n\x10\x62ox_coord_offset\x18\x0c \x01(\x05:\x01\x30\x12\x12\n\x07x_scale\x18\x04 \x01(\x02:\x01\x30\x12\x12\n\x07y_scale\x18\x05 \x01(\x02:\x01\x30\x12\x12\n\x07w_scale\x18\x06 \x01(\x02:\x01\x30\x12\x12\n\x07h_scale\x18\x07 \x01(\x02:\x01\x30\x12,\n\x1d\x61pply_exponential_on_box_size\x18\r \x01(\x08:\x05\x66\x61lse\x12#\n\x14reverse_output_order\x18\x0e \x01(\x08:\x05\x66\x61lse\x12\x16\n\x0eignore_classes\x18\x08 \x03(\x05\x12\x1c\n\rsigmoid_score\x18\x0f \x01(\x08:\x05\x66\x61lse\x12\x1d\n\x15score_clipping_thresh\x18\x10 \x01(\x02\x12\x1e\n\x0f\x66lip_vertically\x18\x12 \x01(\x08:\x05\x66\x61lse\x12\x18\n\x10min_score_thresh\x18\x13 \x01(\x02\x32\x63\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\x98\x8a\xc6u \x01(\x0b\x32\x35.mediapipe.TfLiteTensorsToDetectionsCalculatorOptions')
  ,
  dependencies=[mediapipe_dot_framework_dot_calculator__pb2.DESCRIPTOR,])




_TFLITETENSORSTODETECTIONSCALCULATOROPTIONS = _descriptor.Descriptor(
  name='TfLiteTensorsToDetectionsCalculatorOptions',
  full_name='mediapipe.TfLiteTensorsToDetectionsCalculatorOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_classes', full_name='mediapipe.TfLiteTensorsToDetectionsCalculatorOptions.num_classes', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_boxes', full_name='mediapipe.TfLiteTensorsToDetectionsCalculatorOptions.num_boxes', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_coords', full_name='mediapipe.TfLiteTensorsToDetectionsCalculatorOptions.num_coords', index=2,
      number=3, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='keypoint_coord_offset', full_name='mediapipe.TfLiteTensorsToDetectionsCalculatorOptions.keypoint_coord_offset', index=3,
      number=9, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_keypoints', full_name='mediapipe.TfLiteTensorsToDetectionsCalculatorOptions.num_keypoints', index=4,
      number=10, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_values_per_keypoint', full_name='mediapipe.TfLiteTensorsToDetectionsCalculatorOptions.num_values_per_keypoint', index=5,
      number=11, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=2,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='box_coord_offset', full_name='mediapipe.TfLiteTensorsToDetectionsCalculatorOptions.box_coord_offset', index=6,
      number=12, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='x_scale', full_name='mediapipe.TfLiteTensorsToDetectionsCalculatorOptions.x_scale', index=7,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='y_scale', full_name='mediapipe.TfLiteTensorsToDetectionsCalculatorOptions.y_scale', index=8,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='w_scale', full_name='mediapipe.TfLiteTensorsToDetectionsCalculatorOptions.w_scale', index=9,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='h_scale', full_name='mediapipe.TfLiteTensorsToDetectionsCalculatorOptions.h_scale', index=10,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='apply_exponential_on_box_size', full_name='mediapipe.TfLiteTensorsToDetectionsCalculatorOptions.apply_exponential_on_box_size', index=11,
      number=13, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='reverse_output_order', full_name='mediapipe.TfLiteTensorsToDetectionsCalculatorOptions.reverse_output_order', index=12,
      number=14, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ignore_classes', full_name='mediapipe.TfLiteTensorsToDetectionsCalculatorOptions.ignore_classes', index=13,
      number=8, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sigmoid_score', full_name='mediapipe.TfLiteTensorsToDetectionsCalculatorOptions.sigmoid_score', index=14,
      number=15, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='score_clipping_thresh', full_name='mediapipe.TfLiteTensorsToDetectionsCalculatorOptions.score_clipping_thresh', index=15,
      number=16, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='flip_vertically', full_name='mediapipe.TfLiteTensorsToDetectionsCalculatorOptions.flip_vertically', index=16,
      number=18, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='min_score_thresh', full_name='mediapipe.TfLiteTensorsToDetectionsCalculatorOptions.min_score_thresh', index=17,
      number=19, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
    _descriptor.FieldDescriptor(
      name='ext', full_name='mediapipe.TfLiteTensorsToDetectionsCalculatorOptions.ext', index=0,
      number=246514968, type=11, cpp_type=10, label=1,
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
  serialized_start=128,
  serialized_end=761,
)

DESCRIPTOR.message_types_by_name['TfLiteTensorsToDetectionsCalculatorOptions'] = _TFLITETENSORSTODETECTIONSCALCULATOROPTIONS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TfLiteTensorsToDetectionsCalculatorOptions = _reflection.GeneratedProtocolMessageType('TfLiteTensorsToDetectionsCalculatorOptions', (_message.Message,), dict(
  DESCRIPTOR = _TFLITETENSORSTODETECTIONSCALCULATOROPTIONS,
  __module__ = 'mediapipe.calculators.tflite.tflite_tensors_to_detections_calculator_pb2'
  # @@protoc_insertion_point(class_scope:mediapipe.TfLiteTensorsToDetectionsCalculatorOptions)
  ))
_sym_db.RegisterMessage(TfLiteTensorsToDetectionsCalculatorOptions)

_TFLITETENSORSTODETECTIONSCALCULATOROPTIONS.extensions_by_name['ext'].message_type = _TFLITETENSORSTODETECTIONSCALCULATOROPTIONS
mediapipe_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_TFLITETENSORSTODETECTIONSCALCULATOROPTIONS.extensions_by_name['ext'])

# @@protoc_insertion_point(module_scope)
