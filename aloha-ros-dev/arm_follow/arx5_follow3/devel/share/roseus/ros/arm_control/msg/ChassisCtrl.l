;; Auto-generated. Do not edit!


(when (boundp 'arm_control::ChassisCtrl)
  (if (not (find-package "ARM_CONTROL"))
    (make-package "ARM_CONTROL"))
  (shadow 'ChassisCtrl (find-package "ARM_CONTROL")))
(unless (find-package "ARM_CONTROL::CHASSISCTRL")
  (make-package "ARM_CONTROL::CHASSISCTRL"))

(in-package "ROS")
;;//! \htmlinclude ChassisCtrl.msg.html


(defclass arm_control::ChassisCtrl
  :super ros::object
  :slots (_vx _vy _vz _mode1 _mode2 ))

(defmethod arm_control::ChassisCtrl
  (:init
   (&key
    ((:vx __vx) 0.0)
    ((:vy __vy) 0.0)
    ((:vz __vz) 0.0)
    ((:mode1 __mode1) 0)
    ((:mode2 __mode2) 0)
    )
   (send-super :init)
   (setq _vx (float __vx))
   (setq _vy (float __vy))
   (setq _vz (float __vz))
   (setq _mode1 (round __mode1))
   (setq _mode2 (round __mode2))
   self)
  (:vx
   (&optional __vx)
   (if __vx (setq _vx __vx)) _vx)
  (:vy
   (&optional __vy)
   (if __vy (setq _vy __vy)) _vy)
  (:vz
   (&optional __vz)
   (if __vz (setq _vz __vz)) _vz)
  (:mode1
   (&optional __mode1)
   (if __mode1 (setq _mode1 __mode1)) _mode1)
  (:mode2
   (&optional __mode2)
   (if __mode2 (setq _mode2 __mode2)) _mode2)
  (:serialization-length
   ()
   (+
    ;; float64 _vx
    8
    ;; float64 _vy
    8
    ;; float64 _vz
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
     ;; float64 _vx
       (sys::poke _vx (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _vy
       (sys::poke _vy (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _vz
       (sys::poke _vz (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; int32 _mode1
       (write-long _mode1 s)
     ;; int32 _mode2
       (write-long _mode2 s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float64 _vx
     (setq _vx (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _vy
     (setq _vy (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _vz
     (setq _vz (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; int32 _mode1
     (setq _mode1 (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _mode2
     (setq _mode2 (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get arm_control::ChassisCtrl :md5sum-) "488e9407ad3cab6f3db0659c324cb1ea")
(setf (get arm_control::ChassisCtrl :datatype-) "arm_control/ChassisCtrl")
(setf (get arm_control::ChassisCtrl :definition-)
      "float64 vx
float64 vy
float64 vz
int32 mode1
int32 mode2
")



(provide :arm_control/ChassisCtrl "488e9407ad3cab6f3db0659c324cb1ea")


