;; Auto-generated. Do not edit!


(when (boundp 'arm_control::MagicCmd)
  (if (not (find-package "ARM_CONTROL"))
    (make-package "ARM_CONTROL"))
  (shadow 'MagicCmd (find-package "ARM_CONTROL")))
(unless (find-package "ARM_CONTROL::MAGICCMD")
  (make-package "ARM_CONTROL::MAGICCMD"))

(in-package "ROS")
;;//! \htmlinclude MagicCmd.msg.html


(defclass arm_control::MagicCmd
  :super ros::object
  :slots (_x _y _z _roll _pitch _yaw _gripper _mode1 _mode2 ))

(defmethod arm_control::MagicCmd
  (:init
   (&key
    ((:x __x) 0.0)
    ((:y __y) 0.0)
    ((:z __z) 0.0)
    ((:roll __roll) 0.0)
    ((:pitch __pitch) 0.0)
    ((:yaw __yaw) 0.0)
    ((:gripper __gripper) 0.0)
    ((:mode1 __mode1) 0)
    ((:mode2 __mode2) 0)
    )
   (send-super :init)
   (setq _x (float __x))
   (setq _y (float __y))
   (setq _z (float __z))
   (setq _roll (float __roll))
   (setq _pitch (float __pitch))
   (setq _yaw (float __yaw))
   (setq _gripper (float __gripper))
   (setq _mode1 (round __mode1))
   (setq _mode2 (round __mode2))
   self)
  (:x
   (&optional __x)
   (if __x (setq _x __x)) _x)
  (:y
   (&optional __y)
   (if __y (setq _y __y)) _y)
  (:z
   (&optional __z)
   (if __z (setq _z __z)) _z)
  (:roll
   (&optional __roll)
   (if __roll (setq _roll __roll)) _roll)
  (:pitch
   (&optional __pitch)
   (if __pitch (setq _pitch __pitch)) _pitch)
  (:yaw
   (&optional __yaw)
   (if __yaw (setq _yaw __yaw)) _yaw)
  (:gripper
   (&optional __gripper)
   (if __gripper (setq _gripper __gripper)) _gripper)
  (:mode1
   (&optional __mode1)
   (if __mode1 (setq _mode1 __mode1)) _mode1)
  (:mode2
   (&optional __mode2)
   (if __mode2 (setq _mode2 __mode2)) _mode2)
  (:serialization-length
   ()
   (+
    ;; float64 _x
    8
    ;; float64 _y
    8
    ;; float64 _z
    8
    ;; float64 _roll
    8
    ;; float64 _pitch
    8
    ;; float64 _yaw
    8
    ;; float64 _gripper
    8
    ;; int32 _mode1
    4
    ;; int32 _mode2
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float64 _x
       (sys::poke _x (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _y
       (sys::poke _y (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _z
       (sys::poke _z (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _roll
       (sys::poke _roll (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _pitch
       (sys::poke _pitch (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _yaw
       (sys::poke _yaw (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _gripper
       (sys::poke _gripper (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; int32 _mode1
       (write-long _mode1 s)
     ;; int32 _mode2
       (write-long _mode2 s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float64 _x
     (setq _x (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _y
     (setq _y (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _z
     (setq _z (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _roll
     (setq _roll (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _pitch
     (setq _pitch (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _yaw
     (setq _yaw (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _gripper
     (setq _gripper (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; int32 _mode1
     (setq _mode1 (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _mode2
     (setq _mode2 (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get arm_control::MagicCmd :md5sum-) "1c57ab1c25d2c8b93367daa40ff13cae")
(setf (get arm_control::MagicCmd :datatype-) "arm_control/MagicCmd")
(setf (get arm_control::MagicCmd :definition-)
      "float64 x
float64 y
float64 z
float64 roll
float64 pitch
float64 yaw
float64 gripper
int32 mode1
int32 mode2
")



(provide :arm_control/MagicCmd "1c57ab1c25d2c8b93367daa40ff13cae")


