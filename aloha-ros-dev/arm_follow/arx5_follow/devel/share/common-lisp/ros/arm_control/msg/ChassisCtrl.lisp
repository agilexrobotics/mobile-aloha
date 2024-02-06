; Auto-generated. Do not edit!


(cl:in-package arm_control-msg)


;//! \htmlinclude ChassisCtrl.msg.html

(cl:defclass <ChassisCtrl> (roslisp-msg-protocol:ros-message)
  ((vx
    :reader vx
    :initarg :vx
    :type cl:float
    :initform 0.0)
   (vy
    :reader vy
    :initarg :vy
    :type cl:float
    :initform 0.0)
   (vz
    :reader vz
    :initarg :vz
    :type cl:float
    :initform 0.0)
   (mode1
    :reader mode1
    :initarg :mode1
    :type cl:integer
    :initform 0)
   (mode2
    :reader mode2
    :initarg :mode2
    :type cl:integer
    :initform 0))
)

(cl:defclass ChassisCtrl (<ChassisCtrl>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ChassisCtrl>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ChassisCtrl)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name arm_control-msg:<ChassisCtrl> is deprecated: use arm_control-msg:ChassisCtrl instead.")))

(cl:ensure-generic-function 'vx-val :lambda-list '(m))
(cl:defmethod vx-val ((m <ChassisCtrl>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader arm_control-msg:vx-val is deprecated.  Use arm_control-msg:vx instead.")
  (vx m))

(cl:ensure-generic-function 'vy-val :lambda-list '(m))
(cl:defmethod vy-val ((m <ChassisCtrl>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader arm_control-msg:vy-val is deprecated.  Use arm_control-msg:vy instead.")
  (vy m))

(cl:ensure-generic-function 'vz-val :lambda-list '(m))
(cl:defmethod vz-val ((m <ChassisCtrl>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader arm_control-msg:vz-val is deprecated.  Use arm_control-msg:vz instead.")
  (vz m))

(cl:ensure-generic-function 'mode1-val :lambda-list '(m))
(cl:defmethod mode1-val ((m <ChassisCtrl>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader arm_control-msg:mode1-val is deprecated.  Use arm_control-msg:mode1 instead.")
  (mode1 m))

(cl:ensure-generic-function 'mode2-val :lambda-list '(m))
(cl:defmethod mode2-val ((m <ChassisCtrl>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader arm_control-msg:mode2-val is deprecated.  Use arm_control-msg:mode2 instead.")
  (mode2 m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ChassisCtrl>) ostream)
  "Serializes a message object of type '<ChassisCtrl>"
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'vx))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'vy))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'vz))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let* ((signed (cl:slot-value msg 'mode1)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'mode2)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ChassisCtrl>) istream)
  "Deserializes a message object of type '<ChassisCtrl>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'vx) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'vy) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'vz) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'mode1) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'mode2) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ChassisCtrl>)))
  "Returns string type for a message object of type '<ChassisCtrl>"
  "arm_control/ChassisCtrl")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ChassisCtrl)))
  "Returns string type for a message object of type 'ChassisCtrl"
  "arm_control/ChassisCtrl")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ChassisCtrl>)))
  "Returns md5sum for a message object of type '<ChassisCtrl>"
  "488e9407ad3cab6f3db0659c324cb1ea")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ChassisCtrl)))
  "Returns md5sum for a message object of type 'ChassisCtrl"
  "488e9407ad3cab6f3db0659c324cb1ea")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ChassisCtrl>)))
  "Returns full string definition for message of type '<ChassisCtrl>"
  (cl:format cl:nil "float64 vx~%float64 vy~%float64 vz~%int32 mode1~%int32 mode2~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ChassisCtrl)))
  "Returns full string definition for message of type 'ChassisCtrl"
  (cl:format cl:nil "float64 vx~%float64 vy~%float64 vz~%int32 mode1~%int32 mode2~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ChassisCtrl>))
  (cl:+ 0
     8
     8
     8
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ChassisCtrl>))
  "Converts a ROS message object to a list"
  (cl:list 'ChassisCtrl
    (cl:cons ':vx (vx msg))
    (cl:cons ':vy (vy msg))
    (cl:cons ':vz (vz msg))
    (cl:cons ':mode1 (mode1 msg))
    (cl:cons ':mode2 (mode2 msg))
))
