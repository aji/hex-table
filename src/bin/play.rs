use clap::Parser;
use ggez::{
    Context, ContextBuilder, GameError, GameResult,
    conf::WindowMode,
    event::EventHandler,
    glam::{Affine2, Vec2},
    graphics::{Canvas, Color, DrawMode, DrawParam, Mesh, Rect, Text, TextLayout},
    winit::event::MouseButton,
};
use hex_table::{
    agent::{Agent, ThinkHandle, mcts_thinking_task},
    bb::{Bitboard, BitboardPretty},
};

#[cfg(feature = "candle")]
use std::{path::PathBuf, sync::Arc, time::Instant};

#[cfg(feature = "candle")]
use hex_table::{
    agent::AgentThinker,
    nn::{
        candle::model::{CandleDevice, CandleModel},
        search::search as nn_search,
    },
    util::IteratorExt,
};

#[derive(Parser, Debug)]
struct Cli {
    /// Path to a candle checkpoint. If set, the opponent uses the loaded
    /// model; otherwise an MCTS bot is used.
    #[cfg(feature = "candle")]
    #[arg(long, value_name = "PATH")]
    model: Option<PathBuf>,
}

enum Bot {
    Mcts(Agent<fn(ThinkHandle)>),
    #[cfg(feature = "candle")]
    Nn(Agent<NnThinker>),
}

impl Bot {
    fn think(&self, board: Bitboard, turn: usize) -> ThinkHandle {
        match self {
            Bot::Mcts(a) => a.think(board, turn),
            #[cfg(feature = "candle")]
            Bot::Nn(a) => a.think(board, turn),
        }
    }
}

#[cfg(feature = "candle")]
#[derive(Clone)]
struct NnThinker {
    model: Arc<CandleModel>,
}

#[cfg(feature = "candle")]
impl AgentThinker for NnThinker {
    fn think(self, handle: ThinkHandle) {
        nn_thinking_task(&self.model, handle);
    }
}

#[cfg(feature = "candle")]
fn nn_thinking_task(model: &CandleModel, task: ThinkHandle) {
    let board = task.data().board;
    let start = Instant::now();
    let out = nn_search(model, board, 0.0, 0.0, {
        let task = task.clone();
        move |n: usize| {
            let elapsed = start.elapsed();
            let aborted = {
                let mut data = task.data();
                data.message = Some(format!("{:>8} sims {:>7.1?}", n, elapsed));
                data.aborted
            };
            !aborted && elapsed.as_secs() < 20
        }
    });
    let i = out
        .policy
        .iter()
        .copied()
        .map(|x| x.powi(4))
        .sample_weighted(&mut rand::rng())
        .unwrap();
    task.data().result = Some(board.nth_child(i));
}

#[derive(Debug)]
struct BoardGeom {
    rect: Rect,
    scale: f32,
    rc: Affine2,
    rc_inv: Affine2,
}

const WIDTH: f32 = 500.0;
const HEIGHT: f32 = 300.0;
const BOARD_PAD: f32 = 0.05;

const TEXT_PAD: f32 = 5.0;
const TEXT_SCALE: f32 = 16.0;

impl BoardGeom {
    fn new(rect: impl Into<Rect>) -> BoardGeom {
        let rect = rect.into();

        let (dr, dc) = {
            let th = std::f32::consts::TAU / 12.0;
            (Vec2::from_angle(th), Vec2::from_angle(-th))
        };

        let units_w = (dr * 10.0 + dc * 10.0).x + 1.0;
        let units_h = (dr * 10.0 - dc * 10.0).y + 1.0;

        println!("units={units_w},{units_h}");

        let x0 = rect.left();
        let y0 = rect.top();
        let x1 = rect.right();
        let y1 = rect.bottom();

        println!("rect={x0},{y0};{x1},{y1}");

        let scale = {
            let scale_w = (x1 - x0) / units_w;
            let scale_h = (y1 - y0) / units_h;
            scale_w.min(scale_h)
        };
        let offset = {
            let content_w = scale * units_w;
            let content_h = scale * units_h;
            let pad_x = (x1 - x0 - content_w) / 2.0;
            let pad_y = (y1 - y0 - content_h) / 2.0;
            println!("pad={pad_x},{pad_y}");
            Vec2::new(x0 + pad_x + scale / 2.0, y0 + pad_y + content_h / 2.0)
        };

        let rc = Affine2::from_cols(scale * dr, scale * dc, offset);
        let rc_inv = rc.inverse();

        BoardGeom {
            rect,
            scale,
            rc,
            rc_inv,
        }
    }

    fn rc_to_xy(&self, r: usize, c: usize) -> Vec2 {
        self.rc.transform_point2(Vec2::new(r as f32, c as f32))
    }

    fn xy_to_rc(&self, p: Vec2) -> Option<(usize, usize)> {
        let rc = self.rc_inv.transform_point2(p);
        let r = rc.x.round() as isize;
        let c = rc.y.round() as isize;
        match 0 <= r && r < 11 && 0 <= c && c < 11 {
            true => Some((r as usize, c as usize)),
            false => None,
        }
    }
}

struct MainState {
    bot: Bot,
    bot_task: Option<ThinkHandle>,
    bot_message: String,
    cursor: Option<(usize, usize)>,
    cursor_down: Option<(usize, usize)>,
    turn: usize,
    board: Bitboard,
    board_geom: BoardGeom,
    mesh_grid_edge: Mesh,
    mesh_grid_cell: Mesh,
    mesh_grid_piece: Mesh,
}

impl MainState {
    fn new(ctx: &mut Context, bot: Bot) -> GameResult<MainState> {
        let rad = (std::f32::consts::TAU / 12.0).cos() * 2.0 / 3.0;
        let hex_points: Vec<Vec2> = (0..6)
            .map(|i| {
                let th = i as f32 * std::f32::consts::TAU / 6.0;
                Vec2::new(th.cos() * rad, th.sin() * rad)
            })
            .collect();
        let mesh_grid_edge = Mesh::new_polyline(
            ctx,
            DrawMode::stroke(0.1),
            &hex_points[2..=4],
            Color::from_rgb(255, 255, 255),
        )?;
        let mesh_grid_cell = Mesh::new_polygon(
            ctx,
            DrawMode::stroke(0.1),
            &hex_points[..],
            Color::from_rgb(255, 255, 255),
        )?;
        let mesh_grid_piece = Mesh::new_polygon(
            ctx,
            DrawMode::fill(),
            &hex_points[..],
            Color::from_rgb(255, 255, 255),
        )?;

        let board_pad = WIDTH.min(HEIGHT) * BOARD_PAD;
        let board = BoardGeom::new(Rect::new(
            board_pad,
            board_pad,
            WIDTH - 2.0 * board_pad,
            HEIGHT - 2.0 * board_pad,
        ));

        Ok(MainState {
            bot,
            bot_task: None,
            bot_message: "Idle".into(),
            cursor: None,
            cursor_down: None,
            turn: 0,
            board: Bitboard::new(),
            board_geom: board,
            mesh_grid_edge,
            mesh_grid_cell,
            mesh_grid_piece,
        })
    }

    fn rc(&self, r: usize, c: usize) -> DrawParam {
        DrawParam::default()
            .dest(self.board_geom.rc_to_xy(r, c))
            .scale(Vec2::new(self.board_geom.scale, self.board_geom.scale))
    }
}

const C_BLACK: Color = Color::new(0.11, 0.11, 0.11, 1.0);
const C_RED: Color = Color::new(0.93, 0.0, 0.3, 1.0);
const C_WHITE: Color = Color::new(0.83, 0.83, 0.83, 1.0);
const C_BLUE: Color = Color::new(0.0, 0.6, 0.92, 1.0);
const C_GREEN: Color = Color::new(0.6, 0.92, 0.0, 1.0);
const C_DARK_GREY: Color = Color::new(0.25, 0.25, 0.25, 1.0);
const C_LIGHT_GREY: Color = Color::new(0.45, 0.45, 0.45, 1.0);
const C_ACCENT: Color = C_BLUE;

const COLOR_BG: Color = C_BLACK;
const COLOR_GRID: Color = C_DARK_GREY;
const COLOR_SENTE: Color = C_BLUE;
const COLOR_GOTE: Color = C_RED;

impl EventHandler<Context, GameError> for MainState {
    fn update(&mut self, _ctx: &mut Context) -> GameResult {
        if let Some(ref task) = self.bot_task {
            if let Some(result) = task.result() {
                self.board = result;
                self.turn += 1;
                self.bot_task = None;
            }
        } else if !self.board.sente() && self.board.win().is_none() {
            self.bot_task = Some(self.bot.think(self.board, self.turn));
        }

        if self.board.win().is_some() {
            println!("FINAL:\n{}", BitboardPretty(&self.board));
            self.board = Bitboard::new();
            self.turn = 0;
            if let Some(task) = self.bot_task.take() {
                task.abort();
            }
        }

        if let Some(ref task) = self.bot_task {
            self.bot_message = task.message().unwrap_or("Thinking...".into());
        } else {
            self.bot_message = "Idle".into();
        }

        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        let mut canvas = Canvas::from_frame(ctx, COLOR_BG);

        for r in 0..11 {
            for c in 0..11 {
                if let Some(sente) = self.board.rc(r, c) {
                    let color = match sente {
                        true => COLOR_SENTE,
                        false => COLOR_GOTE,
                    };
                    canvas.draw(&self.mesh_grid_piece, self.rc(r, c).color(color));
                }
                canvas.draw(&self.mesh_grid_cell, self.rc(r, c).color(COLOR_GRID));
            }
        }

        for r in 0..11 {
            let th = std::f32::consts::TAU / 6.0;
            canvas.draw(
                &self.mesh_grid_edge,
                self.rc(r, 0)
                    .color(COLOR_SENTE)
                    .offset(Vec2::new(0.1, 0.0))
                    .rotation(th * -1.0),
            );
            canvas.draw(
                &self.mesh_grid_edge,
                self.rc(r, 10)
                    .color(COLOR_SENTE)
                    .offset(Vec2::new(0.1, 0.0))
                    .rotation(th * 2.0),
            );
        }
        for c in 0..11 {
            let th = std::f32::consts::TAU / 6.0;
            canvas.draw(
                &self.mesh_grid_edge,
                self.rc(0, c)
                    .color(COLOR_GOTE)
                    .offset(Vec2::new(0.1, 0.0))
                    .rotation(th * 1.0),
            );
            canvas.draw(
                &self.mesh_grid_edge,
                self.rc(10, c)
                    .color(COLOR_GOTE)
                    .offset(Vec2::new(0.1, 0.0))
                    .rotation(th * -2.0),
            );
        }

        if let Some((r, c)) = self.cursor {
            let color = match self.board.sente() {
                true => COLOR_SENTE,
                false => COLOR_GOTE,
            };
            canvas.draw(&self.mesh_grid_cell, self.rc(r, c).color(color));
        }

        let _ = {
            let mut t = Text::new(format!("TURN:{:3} BOT:{}", self.turn, self.bot_message));
            t.set_layout(TextLayout::top_left());
            t.set_scale(TEXT_SCALE);
            canvas.draw(&t, Vec2::new(TEXT_PAD, HEIGHT - TEXT_PAD - TEXT_SCALE));
        };

        canvas.finish(ctx)?;
        Ok(())
    }

    fn mouse_button_down_event(
        &mut self,
        _ctx: &mut Context,
        _button: MouseButton,
        x: f32,
        y: f32,
    ) -> GameResult {
        self.cursor_down = self.board_geom.xy_to_rc(Vec2::new(x, y));
        Ok(())
    }

    fn mouse_button_up_event(
        &mut self,
        _ctx: &mut Context,
        _button: MouseButton,
        x: f32,
        y: f32,
    ) -> Result<(), GameError> {
        let cursor = self.board_geom.xy_to_rc(Vec2::new(x, y));
        if cursor == self.cursor_down
            && let Some((r, c)) = cursor
        {
            if self.board.rc(r, c).is_none() {
                self.board = self.board.with_move(r, c);
                self.turn += 1;
            }
        }
        Ok(())
    }

    fn mouse_motion_event(
        &mut self,
        _ctx: &mut Context,
        x: f32,
        y: f32,
        _dx: f32,
        _dy: f32,
    ) -> Result<(), ggez::GameError> {
        self.cursor = self.board_geom.xy_to_rc(Vec2::new(x, y));
        Ok(())
    }
}

fn main() -> GameResult {
    let cli = Cli::parse();
    let bot = build_bot(&cli)?;

    let cb = ContextBuilder::new("github.com/aji/hex-table", "Hex")
        .window_mode(WindowMode::default().dimensions(WIDTH, HEIGHT));
    let (mut ctx, ev) = cb.build()?;
    let state = MainState::new(&mut ctx, bot)?;
    ggez::event::run(ctx, ev, state)
}

#[cfg(feature = "candle")]
fn build_bot(cli: &Cli) -> GameResult<Bot> {
    if let Some(path) = cli.model.as_deref() {
        let device = CandleDevice::default();
        println!("loading model {} on {:?}", path.display(), device);
        let bytes = std::fs::read(path)?;
        let model = CandleModel::load_burn(&bytes, &device)
            .map_err(|e| GameError::CustomError(e.to_string()))?;
        Ok(Bot::Nn(Agent::new(NnThinker {
            model: Arc::new(model),
        })))
    } else {
        Ok(Bot::Mcts(Agent::new(mcts_thinking_task)))
    }
}

#[cfg(not(feature = "candle"))]
fn build_bot(_cli: &Cli) -> GameResult<Bot> {
    Ok(Bot::Mcts(Agent::new(mcts_thinking_task)))
}
