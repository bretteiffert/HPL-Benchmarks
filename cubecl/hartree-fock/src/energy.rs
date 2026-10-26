//! Board-energy measurement through libkernelenergy.so (kernel_energy_capi.cpp).

use std::convert::TryFrom;
use std::ffi::c_void;
use std::ptr;

#[link(name = "kernelenergy")]
unsafe extern "C" {
    fn energy_create(gpu_id: i32, stream: *mut c_void) -> *mut c_void;
    fn energy_destroy(handle: *mut c_void);
    fn energy_start(handle: *mut c_void);
    fn energy_stop(handle: *mut c_void);
    fn energy_joules(handle: *mut c_void) -> f64;
    fn energy_sample_count(handle: *mut c_void) -> i32;
}

pub struct EnergyProfiler {
    handle: *mut c_void,
    running: bool,
}

impl EnergyProfiler {
    pub fn new(device_index: usize) -> Result<Self, String> {
        let gpu_id = i32::try_from(device_index)
            .map_err(|_| "GPU device index does not fit in the C API's int type".to_string())?;
        // The profiler integrates board power independently of the stream. A null
        // stream selects the default stream for its CUDA/HIP events.
        let handle = unsafe { energy_create(gpu_id, ptr::null_mut()) };
        if handle.is_null() {
            return Err("energy_create() failed; rerun with --no-energy to skip energy measurement"
                .to_string());
        }
        Ok(Self { handle, running: false })
    }

    pub fn start(&mut self) {
        unsafe { energy_start(self.handle) };
        self.running = true;
    }

    /// Returns (total joules, sample count).
    pub fn stop_and_read(&mut self) -> (f64, u32) {
        if self.running {
            unsafe { energy_stop(self.handle) };
            self.running = false;
        }
        unsafe {
            (
                energy_joules(self.handle),
                energy_sample_count(self.handle).max(0) as u32,
            )
        }
    }
}

impl Drop for EnergyProfiler {
    fn drop(&mut self) {
        if self.running {
            unsafe { energy_stop(self.handle) };
        }
        unsafe { energy_destroy(self.handle) };
    }
}
