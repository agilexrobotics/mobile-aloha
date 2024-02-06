; Auto-generated. Do not edit!


(cl:in-package arm_control-msg)


;//! \htmlinclude JointInformation.msg.html

(cl:defclass <JointInformation> (roslisp-msg-protocol:ros-message)
  ((joint_pos
    :reader joint_pos
    :initarg :joint_pos
    :type (cl:vector cl:float)
   :initform (cl:make-array 7 :element-type 'cl:float :initial-element 0.0))
   (joint_dev
    :reader joint_dev
    :initarg :joint_dev
    :type (cl:vector cl:float)
   :initform (cl:make-array 7 :element-type 'cl:float :initial-element 0.0))
   (joint_cur
    :reader joint_cur
    :initarg :joint_cur
    :type (cl:vector cl:float)
   :initform (cl:make-array 7 :element-type 'cl:float :initial-element 0.0))
   (mode
    :reader mode
    :initarg :mode
    :type cl:integer
    :initform 0))
)

(cl:defclass JointInformation (<JointInformation>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <JointInformation>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'JointInformation)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name arm_control-msg:<JointInformation> is deprecated: use arm_control-msg:JointInformation instead.")))

(cl:ensure-generic-function 'joint_pos-val :lambda-list '(m))
(cl:defmethod joint_pos-val ((m <JointInformation>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader arm_control-msg:joint_pos-val is deprecated.  Use arm_control-msg:joint_pos instead.")
  (joint_pos m))

(cl:ensure-generic-function 'joint_dev-val :lambda-list '(m))
(cl:defmethod joint_dev-val ((m <JointInformation>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader arm_control-msg:joint_dev-val is deprecated.  Use arm_control-msg:joint_dev instead.")
  (joint_dev m))

(cl:ensure-generic-function 'joint_cur-val :lambda-list '(m))
(cl:defmethod joint_cur-val ((m <JointInformation>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader arm_control-msg:joint_cur-val is deprecated.  Use arm_control-msg:joint_cur instead.")
  (joint_cur m))

(cl:ensure-generic-function 'mode-val :lambda-list '(m))
(cl:defmethod mode-val ((m <JointInformation>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader arm_control-msg:mode-val is deprecated.  Use arm_control-msg:mode instead.")
  (mode m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <JointInformation>) ostream)
  "Serializes a message object of type '<JointInformation>"
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'joint_pos))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'joint_dev))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'joint_cur))
  (cl:let* ((signed (cl:slot-value msg 'mode)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <JointInformation>) istream)
  "Deserializes a message object of type '<JointInformation>"
  (cl:setf (cl:slot-value msg 'joint_pos) (cl:make-array 7))
  (cl:let ((vals (cl:slot-value msg 'joint_pos)))
    (cl:dotimes (i 7)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits)))))
  (cl:setf (cl:slot-value msg 'joint_dev) (cl:make-array 7))
  (cl:let ((vals (cl:slot-value msg 'joint_dev)))
    (cl:dotimes (i 7)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits)))))
  (cl:setf (cl:slot-value msg 'joint_cur) (cl:make-array 7))
  (cl:let ((vals (cl:slot-value msg 'joint_cur)))
    (cl:dotimes (i 7)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits)))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'mode) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<JointInformation>)))
  "Returns string type for a message object of type '<JointInformation>"
  "arm_control/JointInformation")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'JointInformation)))
  "Returns string type for a message object of type 'JointInformation"
  "arm_control/JointInformation")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<JointInformation>)))
  "Returns md5sum for a message object of type '<JointInformation>"
  "07ea87c1a9d81347441927ac49537ad0")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'JointInformation)))
  "Returns md5sum for a message object of type 'JointInformation"
  "07ea87c1a9d81347441927ac49537ad0")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<JointInformation>)))
  "Returns full string definition for message of type '<JointInformation>"
  (cl:format cl:nil "float32[7] joint_pos~%float32[7] joint_dev~%float32[7] joint_cur~%int32 mode~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'JointInformation)))
  "Returns full string definition for message of type 'JointInformation"
  (cl:format cl:nil "float32[7] joint_pos~%float32[7] joint_dev~%float32[7] joint_cur~%int32 mode~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <JointInformation>))
  (cl:+ 0
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'joint_pos) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'joint_dev) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'joint_cur) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <JointInformation>))
  "Converts a ROS message object to a list"
  (cl:list 'JointInformation
    (cl:cons ':joint_pos (joint_pos msg))
    (cl:cons ':joint_dev (joint_dev msg))
    (cl:cons ':joint_cur (joint_cur msg))
    (cl:cons ':mode (mode msg))
))
