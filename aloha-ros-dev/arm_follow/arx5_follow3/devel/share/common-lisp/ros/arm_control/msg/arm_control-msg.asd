
(cl:in-package :asdf)

(defsystem "arm_control-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "ChassisCtrl" :depends-on ("_package_ChassisCtrl"))
    (:file "_package_ChassisCtrl" :depends-on ("_package"))
    (:file "JointControl" :depends-on ("_package_JointControl"))
    (:file "_package_JointControl" :depends-on ("_package"))
    (:file "JointInformation" :depends-on ("_package_JointInformation"))
    (:file "_package_JointInformation" :depends-on ("_package"))
    (:file "MagicCmd" :depends-on ("_package_MagicCmd"))
    (:file "_package_MagicCmd" :depends-on ("_package"))
    (:file "arx5" :depends-on ("_package_arx5"))
    (:file "_package_arx5" :depends-on ("_package"))
  ))