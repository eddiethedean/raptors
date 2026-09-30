//! Python entry point for the checked Raptors 0.3 numeric ufunc preview.
mod preview;
use pyo3::prelude::*;

#[cfg(feature = "alloc-profile")]
mod allocation_profile {
    use pyo3::prelude::*;
    use pyo3::types::PyModule;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    static ENABLED: AtomicBool = AtomicBool::new(false);
    static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
    static ZEROED_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
    static REALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
    static DEALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
    static REQUESTED_BYTES: AtomicUsize = AtomicUsize::new(0);
    static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
    static BASELINE_LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
    static PEAK_LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);

    pub(super) struct TrackingAllocator;

    unsafe impl GlobalAlloc for TrackingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            // SAFETY: The layout is passed through unchanged to the system allocator.
            let pointer = unsafe { System.alloc(layout) };
            if !pointer.is_null() {
                record_allocation(layout.size(), false);
            }
            pointer
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            // SAFETY: The layout is passed through unchanged to the system allocator.
            let pointer = unsafe { System.alloc_zeroed(layout) };
            if !pointer.is_null() {
                record_allocation(layout.size(), true);
            }
            pointer
        }

        unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
            if !pointer.is_null() {
                if ENABLED.load(Ordering::Relaxed) {
                    DEALLOCATIONS.fetch_add(1, Ordering::Relaxed);
                }
                LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
            }
            // SAFETY: The pointer and original layout are passed through unchanged.
            unsafe { System.dealloc(pointer, layout) };
        }

        unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            // SAFETY: The pointer and original layout satisfy GlobalAlloc::realloc's contract.
            let new_pointer = unsafe { System.realloc(pointer, layout, new_size) };
            if !new_pointer.is_null() {
                if ENABLED.load(Ordering::Relaxed) {
                    REALLOCATIONS.fetch_add(1, Ordering::Relaxed);
                    REQUESTED_BYTES.fetch_add(new_size, Ordering::Relaxed);
                }
                if new_size >= layout.size() {
                    let increase = new_size - layout.size();
                    let live = LIVE_BYTES.fetch_add(increase, Ordering::Relaxed) + increase;
                    if ENABLED.load(Ordering::Relaxed) {
                        record_peak(live);
                    }
                } else {
                    LIVE_BYTES.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
                }
            }
            new_pointer
        }
    }

    fn record_allocation(bytes: usize, zeroed: bool) {
        let live = LIVE_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
        if ENABLED.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            if zeroed {
                ZEROED_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            }
            REQUESTED_BYTES.fetch_add(bytes, Ordering::Relaxed);
            record_peak(live);
        }
    }

    fn record_peak(live: usize) {
        let baseline = BASELINE_LIVE_BYTES.load(Ordering::Relaxed);
        let delta = live.saturating_sub(baseline);
        let mut peak = PEAK_LIVE_BYTES.load(Ordering::Relaxed);
        while delta > peak {
            match PEAK_LIVE_BYTES.compare_exchange_weak(
                peak,
                delta,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(current) => peak = current,
            }
        }
    }

    #[pyfunction]
    fn start() {
        ENABLED.store(false, Ordering::SeqCst);
        ALLOCATIONS.store(0, Ordering::Relaxed);
        ZEROED_ALLOCATIONS.store(0, Ordering::Relaxed);
        REALLOCATIONS.store(0, Ordering::Relaxed);
        DEALLOCATIONS.store(0, Ordering::Relaxed);
        REQUESTED_BYTES.store(0, Ordering::Relaxed);
        let baseline = LIVE_BYTES.load(Ordering::Relaxed);
        BASELINE_LIVE_BYTES.store(baseline, Ordering::Relaxed);
        PEAK_LIVE_BYTES.store(0, Ordering::Relaxed);
        ENABLED.store(true, Ordering::SeqCst);
    }

    #[pyfunction]
    fn stop() -> (usize, usize, usize, usize, usize, usize, usize) {
        ENABLED.store(false, Ordering::SeqCst);
        let live = LIVE_BYTES.load(Ordering::Relaxed);
        let baseline = BASELINE_LIVE_BYTES.load(Ordering::Relaxed);
        (
            ALLOCATIONS.load(Ordering::Relaxed),
            ZEROED_ALLOCATIONS.load(Ordering::Relaxed),
            REALLOCATIONS.load(Ordering::Relaxed),
            DEALLOCATIONS.load(Ordering::Relaxed),
            REQUESTED_BYTES.load(Ordering::Relaxed),
            PEAK_LIVE_BYTES.load(Ordering::Relaxed),
            live.saturating_sub(baseline),
        )
    }

    pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
        let benchmark_module = PyModule::new(module.py(), "_bench")?;
        benchmark_module.add_function(wrap_pyfunction!(start, &benchmark_module)?)?;
        benchmark_module.add_function(wrap_pyfunction!(stop, &benchmark_module)?)?;
        module.add_submodule(&benchmark_module)
    }
}

#[cfg(feature = "alloc-profile")]
#[global_allocator]
static GLOBAL_ALLOCATOR: allocation_profile::TrackingAllocator =
    allocation_profile::TrackingAllocator;

#[pymodule]
fn raptors(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    preview::register(module)?;
    #[cfg(feature = "alloc-profile")]
    allocation_profile::register(module)?;
    Ok(())
}
