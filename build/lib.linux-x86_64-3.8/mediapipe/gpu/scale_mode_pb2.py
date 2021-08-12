# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/gpu/scale_mode.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='mediapipe/gpu/scale_mode.proto',
  package='mediapipe',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n\x1emediapipe/gpu/scale_mode.proto\x12\tmediapipe\"I\n\tScaleMode\"<\n\x04Mode\x12\x0b\n\x07\x44\x45\x46\x41ULT\x10\x00\x12\x0b\n\x07STRETCH\x10\x01\x12\x07\n\x03\x46IT\x10\x02\x12\x11\n\rFILL_AND_CROP\x10\x03')
)



_SCALEMODE_MODE = _descriptor.EnumDescriptor(
  name='Mode',
  full_name='mediapipe.ScaleMode.Mode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='DEFAULT', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='STRETCH', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FIT', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FILL_AND_CROP', index=3, number=3,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=58,
  serialized_end=118,
)
_sym_db.RegisterEnumDescriptor(_SCALEMODE_MODE)


_SCALEMODE = _descriptor.Descriptor(
  name='ScaleMode',
  full_name='mediapipe.ScaleMode',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _SCALEMODE_MODE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=45,
  serialized_end=118,
)

_SCALEMODE_MODE.containing_type = _SCALEMODE
DESCRIPTOR.message_types_by_name['ScaleMode'] = _SCALEMODE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ScaleMode = _reflection.GeneratedProtocolMessageType('ScaleMode', (_message.Message,), dict(
  DESCRIPTOR = _SCALEMODE,
  __module__ = 'mediapipe.gpu.scale_mode_pb2'
  # @@protoc_insertion_point(class_scope:mediapipe.ScaleMode)
  ))
_sym_db.RegisterMessage(ScaleMode)


# @@protoc_insertion_point(module_scope)
