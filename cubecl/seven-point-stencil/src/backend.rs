//! Build-time backend selection plus device metadata.
//!
//! CubeCL 0.10 does not report the device name; the
//! SM/CU count for both backends, and enumerates devices. The one thing it does
//! not report is the driver version, so that alone is queried from the vendor
//! driver library, loaded at runtime. The query is best-effort: on any failure
//! the driver shows as "unknown".

#[cfg(all(feature = "cuda", feature = "hip"))]
compile_error!("Enable exactly one backend feature: `cuda` or `hip`, not both.");
#[cfg(not(any(feature = "cuda", feature = "hip")))]
compile_error!("Enable a backend feature: `--features cuda` or `--features hip`.");

use cubecl::prelude::{ComputeClient, Runtime};
use libloading::Library;

#[cfg(feature = "cuda")]
pub type BackendRuntime = cubecl::cuda::CudaRuntime;
#[cfg(feature = "cuda")]
pub type BackendDevice = cubecl::cuda::CudaDevice;
#[cfg(all(feature = "hip", not(feature = "cuda")))]
pub type BackendRuntime = cubecl::hip::HipRuntime;
#[cfg(all(feature = "hip", not(feature = "cuda")))]
pub type BackendDevice = cubecl::hip::AmdDevice;
pub type Client = ComputeClient<BackendRuntime>;

#[cfg(feature = "cuda")]
pub const BACKEND_LABEL: &str = "CUDA";
#[cfg(all(feature = "hip", not(feature = "cuda")))]
pub const BACKEND_LABEL: &str = "HIP";

/// The device at `index`, or an error naming how many this machine has.
pub fn device(index: usize) -> Result<BackendDevice, String> {
    let count = vendor::device_count().unwrap_or(0);
    if index >= count { return Err(format!("device index {} out of range ({} device(s) available)", index, count)); }
    #[cfg(feature = "cuda")]
    let device = BackendDevice::new(index);
    #[cfg(all(feature = "hip", not(feature = "cuda")))]
    let device = BackendDevice::new(index);
    Ok(device)
}

/// Every device this backend can see, in index order.
pub fn devices() -> Vec<BackendDevice> {
    (0..).map_while(|i| device(i).ok()).collect()
}

pub fn client(device: &BackendDevice) -> Client { BackendRuntime::client(device) }

/// Placeholder used when the device reports no name.
pub fn placeholder_name(index: usize) -> String {
    format!("{} device {}", BACKEND_LABEL, index)
}

/// Device name: `GPU_NAME` override, then the device's own name, then a placeholder.
pub fn resolve_name(index: usize, _client: &Client) -> (String, Option<String>) {
    if let Ok(name) = std::env::var("GPU_NAME") {
        return (name, None);
    }
    (placeholder_name(index), Some("CubeCL 0.10 does not report device names; set GPU_NAME to label results".to_string()))
}

/// Driver string: `GPU_DRIVER` override, then the driver query, then "unknown".
pub fn resolve_driver() -> String {
    std::env::var("GPU_DRIVER")
        .ok()
        .or_else(vendor::driver_version)
        .unwrap_or_else(|| "unknown".to_string())
}

fn open_first(candidates: &[String]) -> Option<Library> {
    candidates
        .iter()
        .find_map(|name| unsafe { Library::new(name.as_str()) }.ok())
}

// ---------------------------------------------------------------------------
// CUDA: NVML for the driver release, libcuda for the CUDA version it supports.
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
mod vendor {
    use super::*;
    use std::ffi::{c_char, CStr};

    fn c_string(buf: &[c_char]) -> Option<String> {
        let s = unsafe { CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .trim()
            .to_string();
        (!s.is_empty()).then_some(s)
    }

    type CuInitFn = unsafe extern "C" fn(u32) -> i32;
    type CuVersionFn = unsafe extern "C" fn(*mut i32) -> i32;
    type CuCountFn = unsafe extern "C" fn(*mut i32) -> i32;
    type NvmlInitFn = unsafe extern "C" fn() -> u32;
    type NvmlDriverFn = unsafe extern "C" fn(*mut c_char, u32) -> u32;
    type NvmlShutdownFn = unsafe extern "C" fn() -> u32;

    pub fn device_count() -> Option<usize> {
        let lib = open_first(&["libcuda.so.1".into(), "libcuda.so".into()])?;
        unsafe {
            let init = lib.get::<CuInitFn>(c"cuInit").ok()?;
            let count = lib.get::<CuCountFn>(c"cuDeviceGetCount").ok()?;
            if init(0) != 0 { return None; }
            let mut value = 0i32;
            (count(&mut value) == 0 && value >= 0).then_some(value as usize)
        }
    }

    /// e.g. "550.54.15 (CUDA 12.4)".
    pub fn driver_version() -> Option<String> {
        match (nvml_driver_release(), cuda_version()) {
            (Some(rel), Some(cuda)) => Some(format!("{} ({})", rel, cuda)),
            (Some(rel), None) => Some(rel),
            (None, cuda) => cuda,
        }
    }

    fn cuda_version() -> Option<String> {
        let lib = open_first(&["libcuda.so.1".into(), "libcuda.so".into()])?;
        unsafe {
            let init = lib.get::<CuInitFn>(c"cuInit").ok()?;
            let version = lib.get::<CuVersionFn>(c"cuDriverGetVersion").ok()?;
            if init(0) != 0 {
                return None;
            }
            let mut v = 0i32;
            (version(&mut v) == 0).then(|| format!("CUDA {}.{}", v / 1000, (v % 1000) / 10))
        }
    }

    fn nvml_driver_release() -> Option<String> {
        let lib = open_first(&["libnvidia-ml.so.1".into(), "libnvidia-ml.so".into()])?;
        unsafe {
            let init = lib.get::<NvmlInitFn>(c"nvmlInit_v2").ok()?;
            let driver = lib.get::<NvmlDriverFn>(c"nvmlSystemGetDriverVersion").ok()?;
            let shutdown = lib.get::<NvmlShutdownFn>(c"nvmlShutdown").ok()?;
            if init() != 0 {
                return None;
            }
            let mut buf = [0 as c_char; 96];
            let ok = driver(buf.as_mut_ptr(), buf.len() as u32) == 0;
            shutdown();
            if ok {
                c_string(&buf)
            } else {
                None
            }
        }
    }
}

// ---------------------------------------------------------------------------
// HIP: runtime library (libamdhip64).
// ---------------------------------------------------------------------------

#[cfg(all(feature = "hip", not(feature = "cuda")))]
mod vendor {
    use super::*;

    type InitFn = unsafe extern "C" fn(u32) -> i32;
    type VersionFn = unsafe extern "C" fn(*mut i32) -> i32;
    type CountFn = unsafe extern "C" fn(*mut i32) -> i32;

    pub fn device_count() -> Option<usize> {
        let mut candidates = Vec::new();
        for var in ["ROCM_PATH", "HIP_PATH"] { if let Ok(root) = std::env::var(var) { candidates.push(format!("{}/lib/libamdhip64.so", root)); } }
        for suffix in ["", ".7", ".6", ".5"] { candidates.push(format!("libamdhip64.so{}", suffix)); }
        let lib = open_first(&candidates)?;
        unsafe {
            let init = lib.get::<InitFn>(c"hipInit").ok()?;
            let count = lib.get::<CountFn>(c"hipGetDeviceCount").ok()?;
            if init(0) != 0 { return None; }
            let mut value = 0i32;
            (count(&mut value) == 0 && value >= 0).then_some(value as usize)
        }
    }

    /// e.g. "HIP 6.3.42134".
    pub fn driver_version() -> Option<String> {
        let mut candidates = Vec::new();
        for var in ["ROCM_PATH", "HIP_PATH"] {
            if let Ok(root) = std::env::var(var) {
                candidates.push(format!("{}/lib/libamdhip64.so", root));
            }
        }
        for suffix in ["", ".7", ".6", ".5"] {
            candidates.push(format!("libamdhip64.so{}", suffix));
        }
        let lib = open_first(&candidates)?;
        unsafe {
            let init = lib.get::<InitFn>(c"hipInit").ok()?;
            let version = lib.get::<VersionFn>(c"hipDriverGetVersion").ok()?;
            if init(0) != 0 {
                return None;
            }
            let mut v = 0i32;
            // HIP_VERSION = major * 10^7 + minor * 10^5 + patch
            (version(&mut v) == 0 && v > 0).then(|| {
                format!(
                    "HIP {}.{}.{}",
                    v / 10_000_000,
                    (v / 100_000) % 100,
                    v % 100_000
                )
            })
        }
    }
}
