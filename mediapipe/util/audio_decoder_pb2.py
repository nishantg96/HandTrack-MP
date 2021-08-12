# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/util/audio_decoder.proto

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
  name='mediapipe/util/audio_decoder.proto',
  package='mediapipe',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n\"mediapipe/util/audio_decoder.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\"\xc1\x01\n\x12\x41udioStreamOptions\x12\x17\n\x0cstream_index\x18\x01 \x01(\x03:\x01\x30\x12\x1c\n\rallow_missing\x18\x02 \x01(\x08:\x05\x66\x61lse\x12%\n\x16ignore_decode_failures\x18\x03 \x01(\x08:\x05\x66\x61lse\x12+\n\x1coutput_regressing_timestamps\x18\x04 \x01(\x08:\x05\x66\x61lse\x12 \n\x18\x63orrect_pts_for_rollover\x18\x05 \x01(\x08\"\xbe\x01\n\x13\x41udioDecoderOptions\x12\x33\n\x0c\x61udio_stream\x18\x01 \x03(\x0b\x32\x1d.mediapipe.AudioStreamOptions\x12\x12\n\nstart_time\x18\x02 \x01(\x01\x12\x10\n\x08\x65nd_time\x18\x03 \x01(\x01\x32L\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\xb2\xef\xca} \x01(\x0b\x32\x1e.mediapipe.AudioDecoderOptions')
  ,
  dependencies=[mediapipe_dot_framework_dot_calculator__pb2.DESCRIPTOR,])




_AUDIOSTREAMOPTIONS = _descriptor.Descriptor(
  name='AudioStreamOptions',
  full_name='mediapipe.AudioStreamOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='stream_index', full_name='mediapipe.AudioStreamOptions.stream_index', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='allow_missing', full_name='mediapipe.AudioStreamOptions.allow_missing', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ignore_decode_failures', full_name='mediapipe.AudioStreamOptions.ignore_decode_failures', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='output_regressing_timestamps', full_name='mediapipe.AudioStreamOptions.output_regressing_timestamps', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='correct_pts_for_rollover', full_name='mediapipe.AudioStreamOptions.correct_pts_for_rollover', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
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
  serialized_start=88,
  serialized_end=281,
)


_AUDIODECODEROPTIONS = _descriptor.Descriptor(
  name='AudioDecoderOptions',
  full_name='mediapipe.AudioDecoderOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='audio_stream', full_name='mediapipe.AudioDecoderOptions.audio_stream', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='start_time', full_name='mediapipe.AudioDecoderOptions.start_time', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='end_time', full_name='mediapipe.AudioDecoderOptions.end_time', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
    _descriptor.FieldDescriptor(
      name='ext', full_name='mediapipe.AudioDecoderOptions.ext', index=0,
      number=263370674, type=11, cpp_type=10, label=1,
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
  serialized_start=284,
  serialized_end=474,
)

_AUDIODECODEROPTIONS.fields_by_name['audio_stream'].message_type = _AUDIOSTREAMOPTIONS
DESCRIPTOR.message_types_by_name['AudioStreamOptions'] = _AUDIOSTREAMOPTIONS
DESCRIPTOR.message_types_by_name['AudioDecoderOptions'] = _AUDIODECODEROPTIONS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

AudioStreamOptions = _reflection.GeneratedProtocolMessageType('AudioStreamOptions', (_message.Message,), dict(
  DESCRIPTOR = _AUDIOSTREAMOPTIONS,
  __module__ = 'mediapipe.util.audio_decoder_pb2'
  # @@protoc_insertion_point(class_scope:mediapipe.AudioStreamOptions)
  ))
_sym_db.RegisterMessage(AudioStreamOptions)

AudioDecoderOptions = _reflection.GeneratedProtocolMessageType('AudioDecoderOptions', (_message.Message,), dict(
  DESCRIPTOR = _AUDIODECODEROPTIONS,
  __module__ = 'mediapipe.util.audio_decoder_pb2'
  # @@protoc_insertion_point(class_scope:mediapipe.AudioDecoderOptions)
  ))
_sym_db.RegisterMessage(AudioDecoderOptions)

_AUDIODECODEROPTIONS.extensions_by_name['ext'].message_type = _AUDIODECODEROPTIONS
mediapipe_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_AUDIODECODEROPTIONS.extensions_by_name['ext'])

# @@protoc_insertion_point(module_scope)
