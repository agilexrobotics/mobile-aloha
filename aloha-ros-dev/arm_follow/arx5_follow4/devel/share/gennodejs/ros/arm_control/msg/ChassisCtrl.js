// Auto-generated. Do not edit!

// (in-package arm_control.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class ChassisCtrl {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.vx = null;
      this.vy = null;
      this.vz = null;
      this.mode1 = null;
      this.mode2 = null;
    }
    else {
      if (initObj.hasOwnProperty('vx')) {
        this.vx = initObj.vx
      }
      else {
        this.vx = 0.0;
      }
      if (initObj.hasOwnProperty('vy')) {
        this.vy = initObj.vy
      }
      else {
        this.vy = 0.0;
      }
      if (initObj.hasOwnProperty('vz')) {
        this.vz = initObj.vz
      }
      else {
        this.vz = 0.0;
      }
      if (initObj.hasOwnProperty('mode1')) {
        this.mode1 = initObj.mode1
      }
      else {
        this.mode1 = 0;
      }
      if (initObj.hasOwnProperty('mode2')) {
        this.mode2 = initObj.mode2
      }
      else {
        this.mode2 = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ChassisCtrl
    // Serialize message field [vx]
    bufferOffset = _serializer.float64(obj.vx, buffer, bufferOffset);
    // Serialize message field [vy]
    bufferOffset = _serializer.float64(obj.vy, buffer, bufferOffset);
    // Serialize message field [vz]
    bufferOffset = _serializer.float64(obj.vz, buffer, bufferOffset);
    // Serialize message field [mode1]
    bufferOffset = _serializer.int32(obj.mode1, buffer, bufferOffset);
    // Serialize message field [mode2]
    bufferOffset = _serializer.int32(obj.mode2, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ChassisCtrl
    let len;
    let data = new ChassisCtrl(null);
    // Deserialize message field [vx]
    data.vx = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [vy]
    data.vy = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [vz]
    data.vz = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [mode1]
    data.mode1 = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [mode2]
    data.mode2 = _deserializer.int32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 32;
  }

  static datatype() {
    // Returns string type for a message object
    return 'arm_control/ChassisCtrl';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '488e9407ad3cab6f3db0659c324cb1ea';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float64 vx
    float64 vy
    float64 vz
    int32 mode1
    int32 mode2
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new ChassisCtrl(null);
    if (msg.vx !== undefined) {
      resolved.vx = msg.vx;
    }
    else {
      resolved.vx = 0.0
    }

    if (msg.vy !== undefined) {
      resolved.vy = msg.vy;
    }
    else {
      resolved.vy = 0.0
    }

    if (msg.vz !== undefined) {
      resolved.vz = msg.vz;
    }
    else {
      resolved.vz = 0.0
    }

    if (msg.mode1 !== undefined) {
      resolved.mode1 = msg.mode1;
    }
    else {
      resolved.mode1 = 0
    }

    if (msg.mode2 !== undefined) {
      resolved.mode2 = msg.mode2;
    }
    else {
      resolved.mode2 = 0
    }

    return resolved;
    }
};

module.exports = ChassisCtrl;
