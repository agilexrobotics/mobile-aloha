;; Auto-generated. Do not edit!


(when (boundp 'arm_control::JointControl)
  (if (not (find-package "ARM_CONTROL"))
    (make-package "ARM_CONTROL"))
  (shadow 'JointControl (find-package "ARM_CONTROL")))
(unless (find-package "ARM_CONTROL::JOINTCONTROL")
  (make-package "ARM_CONTROL::JOINTCONTROL"))

(in-package "ROS")
;;//! \htmlinclude JointControl.msg.html


(defclass arm_control::JointControl
  :super ros::object
  :slots (_joint_pos _joint_dev _joint_cur _mode ))

(defmethod arm_control::JointControl
  (:init
   (&key
    ((:joint_pos __joint_pos) (make-array 7 :initial-element 0.0 :element-type :float))
    ((:joint_dev __joint_dev) (make-array 7 :initial-element 0.0 :element-type :float))
    ((:joint_cur __joint_cur) (make-array 7 :initial-element 0.0 :element-type :float))
    ((:mode __mode) 0)
    )
   (send-super :init)
   (setq _joint_pos __joint_pos)
   (setq _joint_dev __joint_dev)
   (setq _joint_cur __joint_cur)
   (setq _mode (round __mode))
   self)
  (:joint_pos
   (&optional __joint_pos)
   (if __joint_pos (setq _joint_pos __joint_pos)) _joint_pos)
  (:joint_dev
   (&optional __joint_dev)
   (if __joint_dev (setq _joint_dev __joint_dev)) _joint_dev)
  (:joint_cur
   (&optional __joint_cur)
   (if __joint_cur (setq _joint_cur __joint_cur)) _joint_cur)
  (:mode
   (&optional __mode)
   (if __mode (setq _mode __mode)) _mode)
  (:serialization-length
   ()
   (+
    ;; float32[7] _joint_pos
    (* 4    7)
    ;; float32[7] _joint_dev
    (* 4    7)
    ;; float32[7] _joint_cur
    (* 4    7)
    ;; int32 _mode
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32[7] _joint_pos
     (dotimes (i 7)
       (sys::poke (elt _joint_pos i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32[7] _joint_dev
     (dotimes (i 7)
       (sys::poke (elt _joint_dev i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32[7] _joint_cur
     (dotimes (i 7)
       (sys::poke (elt _joint_cur i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; int32 _mode
       (write-long _mode s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32[7] _joint_pos
   (dotimes (i (length _joint_pos))
     (setf (elt _joint_pos i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     )
   ;; float32[7] _joint_dev
   (dotimes (i (length _joint_dev))
     (setf (elt _joint_dev i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     )
   ;; float32[7] _joint_cur
   (dotimes (i (length _joint_cur))
     (setf (elt _joint_cur i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     )
   ;; int32 _mode
     (setq _mode (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get arm_control::JointControl :md5sum-) "07ea87c1a9d81347441927ac49537ad0")
(setf (get arm_control::JointControl :datatype-) "arm_control/JointControl")
(setf (get arm_control::JointControl :definition-)
      "float32[7] joint_pos
float32[7] joint_dev
float32[7] joint_cur
int32 mode
")



(provide :arm_control/JointControl "07ea87c1a9d81347441927ac49537ad0")


