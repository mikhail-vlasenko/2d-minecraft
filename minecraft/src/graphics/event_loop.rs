use egui_winit::winit::{
    event::*,
    event_loop::{EventLoop},
    window::{WindowBuilder},
};
use egui_winit::winit::dpi::PhysicalSize;
use game_logic::SETTINGS;
use crate::graphics::state::State;


pub async fn run() {
    let initial_size = PhysicalSize {
        width: SETTINGS.read().unwrap().window.width,
        height: SETTINGS.read().unwrap().window.height
    };
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("Minecraft")
        .with_inner_size(initial_size)
        .build(&event_loop)
        .unwrap();

    let mut state = State::new(&window).await;

    let _ = event_loop.run(move |event, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == state.window().id() => {
            if !state.input(event) {
                match event {
                    WindowEvent::CloseRequested => control_flow.exit(),
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::RedrawRequested => {
                        // // This tells winit that we want another frame after this one
                        // state.window().request_redraw();

                        state.update();
                        match state.render() {
                            Ok(_) => {}
                            // Reconfigure the surface if it's lost or outdated
                            Err(
                                wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated,
                            ) => state.resize(state.get_size()),
                            // The system is out of memory, we should probably quit
                            Err(wgpu::SurfaceError::OutOfMemory) => {
                                log::error!("OutOfMemory");
                                control_flow.exit();
                            }

                            // This happens when the a frame takes too long to present
                            Err(wgpu::SurfaceError::Timeout) => {
                                log::warn!("Surface timeout")
                            }
                        }
                    }
                    _ => {}
                };
                state.egui_renderer.handle_input(&mut state.window, &event);
            }
        }
        _ => {}
    });
}
