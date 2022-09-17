use enumn::N;

use std::fmt::{self, Display, Formatter};
use std::str::FromStr;

use crate::errors::UnsupportedDeviceError;
use crate::ffi;

/// DLPack device type. See [DLDeviceType](https://dmlc.github.io/dlpack/latest/c_api.html#_CPPv412DLDeviceType)
///
/// ## Example
///
/// ```
/// use dlpackrs::DeviceType;
/// let cpu = DeviceType::from("cpu");
/// println!("device is: {}", cpu);
///```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, N)]
#[repr(u32)]
pub enum DeviceType {
    CPU = 1,
    CUDA = 2,
    CUDAHost = 3,
    OpenCL = 4,
    Vulkan = 7,
    Metal = 8,
    VPI = 9,
    ROCM = 10,
    ROCMHost = 11,
    ExtDev = 12,
    CUDAManaged = 13,
    OneAPI = 14,
    WebGPU = 15,
    Hexagon = 16,
}

impl Default for DeviceType {
    /// default device is cpu.
    fn default() -> Self {
        DeviceType::CPU
    }
}

impl From<DeviceType> for ffi::DLDeviceType {
    fn from(device_type: DeviceType) -> Self {
        device_type as Self
    }
}

impl From<ffi::DLDeviceType> for DeviceType {
    fn from(device_type: ffi::DLDeviceType) -> Self {
        Self::n(device_type as _).expect("invalid enumeration value for ffi::DLDeviceType")
    }
}

impl Display for DeviceType {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                DeviceType::CPU => "cpu",
                DeviceType::CUDA => "cuda",
                DeviceType::CUDAHost => "cuda_host",
                DeviceType::OpenCL => "opencl",
                DeviceType::Vulkan => "vulkan",
                DeviceType::Metal => "metal",
                DeviceType::VPI => "vpi",
                DeviceType::ROCM => "rocm",
                DeviceType::ROCMHost => "rocm_host",
                DeviceType::ExtDev => "ext_device",
                DeviceType::CUDAManaged => "cuda_managed",
                DeviceType::OneAPI => "one_api",
                DeviceType::WebGPU => "web_gpu",
                DeviceType::Hexagon => "hexagon",
            }
        )
    }
}

impl<'a> From<&'a str> for DeviceType {
    fn from(type_str: &'a str) -> Self {
        match type_str {
            "cpu" => DeviceType::CPU,
            "cuda" => DeviceType::CUDA,
            "cuda_host" => DeviceType::CUDAHost,
            "opencl" => DeviceType::OpenCL,
            "vulkan" => DeviceType::Vulkan,
            "metal" => DeviceType::Metal,
            "vpi" => DeviceType::VPI,
            "rocm" => DeviceType::ROCM,
            "rocm_host" => DeviceType::ROCMHost,
            "ext_dev" => DeviceType::ExtDev,
            "cuda_managed" => DeviceType::CUDAManaged,
            "one_api" => DeviceType::OneAPI,
            "web_gpu" => DeviceType::WebGPU,
            "hexagon" => DeviceType::Hexagon,
            _ => panic!("{:?} not supported!", type_str),
        }
    }
}

/// DLPack Device. See [DLDevice](https://dmlc.github.io/dlpack/latest/c_api.html#_CPPv48DLDevice)
///
/// ## Example
///
/// ```
/// use dlpackrs::Device;
/// let dev = Device::cpu(0);
/// println!("device: {}", dev);
/// ```
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[repr(C)]
pub struct Device {
    pub device_type: DeviceType,
    pub device_id: usize,
}

impl Device {
    pub fn new(device_type: DeviceType, device_id: usize) -> Device {
        Device {
            device_type,
            device_id,
        }
    }
}

impl<'a> From<&'a Device> for ffi::DLDevice {
    fn from(dev: &'a Device) -> Self {
        Self {
            device_type: dev.device_type.into(),
            device_id: dev.device_id as i32,
        }
    }
}

impl Default for Device {
    fn default() -> Self {
        Self {
            device_type: ffi::DLDeviceType_kDLCPU.into(),
            device_id: 0,
        }
    }
}

impl<'a> From<&'a str> for Device {
    fn from(target: &str) -> Self {
        Device::new(DeviceType::from(target), 0)
    }
}

impl From<ffi::DLDevice> for Device {
    fn from(dev: ffi::DLDevice) -> Self {
        Device {
            device_type: DeviceType::from(dev.device_type),
            device_id: dev.device_id as usize,
        }
    }
}

impl From<Device> for ffi::DLDevice {
    fn from(dev: Device) -> Self {
        ffi::DLDevice {
            device_type: dev.device_type.into(),
            device_id: dev.device_id as i32,
        }
    }
}

impl Display for Device {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}({})", self.device_type, self.device_id)
    }
}

macro_rules! add_device {
    ( $( $dev_type:ident : [ $( $dev_name:ident ),+ ] ),+ ) => {
        /// Creates a Device from a string (e.g., "cpu", "cuda")
        use DeviceType::*;
        impl FromStr for Device {
            type Err = UnsupportedDeviceError;
            fn from_str(type_str: &str) -> Result<Self, Self::Err> {
                Ok(Self {
                    device_type: match type_str {
                         $( $(  stringify!($dev_name)  )|+ => $dev_type.into()),+,
                        _ => return Err(UnsupportedDeviceError(type_str.to_string())),
                    },
                    device_id: 0,
                })
            }
        }

        impl Device {
            $(
                $(
                    pub fn $dev_name(device_id: usize) -> Self {
                        Self {
                            device_type: $dev_type.into(),
                            device_id,
                        }
                    }
                )+
            )+
        }
    };
}

add_device!(
    CPU: [cpu],
    CUDA: [cuda, nvptx],
    CUDAHost: [cuda_host], // pinned cuda cpu memory
    OpenCL: [cl],
    Vulkan: [vulkan],
    Metal: [metal],
    VPI: [vpi],
    ROCM: [rocm],
    ROCMHost: [rocm_host], // pinned rocm cpu memory
    ExtDev: [ext_dev],
    CUDAManaged: [cuda_managed],
    OneAPI: [one_api],
    WebGPU: [web_gpu],
    Hexagon: [hexagon]
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device() {
        let dev = Device::default();
        println!("device: {}", dev);
        let default_dev = Device::new(DeviceType::CPU, 0);
        assert_eq!(dev.clone(), default_dev);
        assert_ne!(dev, Device::cuda(0));

        let str_dev = Device::new(DeviceType::CUDA, 0);
        assert_eq!(str_dev.clone(), str_dev);
        assert_ne!(str_dev, Device::new(DeviceType::CPU, 0));
    }
}
