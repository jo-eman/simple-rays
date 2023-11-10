extern crate image;
extern crate num_cpus;
extern crate scoped_threadpool;
extern crate stackvec;

#[macro_use]
mod vector;
mod line;
mod shapes;
mod surface;

use obj::{load_obj, Obj};
use std::io::BufReader;
use std::path::Path;
use std::{fs::File, io::BufRead};

use scoped_threadpool::Pool;
use stackvec::prelude::*;

use line::Line;
use shapes::*;
use surface::{Sphere, Triangle};
use vector::{Point, Vector};

// ========================== Float & Wrapper ======================================================

const FLOAT_EPS: f64 = 1e-8;

pub fn is_zero(f: f64) -> bool {
    f.abs() <= FLOAT_EPS
}

#[derive(Copy, Clone, Debug)]
pub enum MathError {
    CollinearVectors,
}

pub type MathResult<T> = Result<T, MathError>;

// ========================== Color & Environment ==================================================

type Color = [u8; 3];

struct ColoredSurface {
    triangle: Triangle,
    color: Color,
}

struct Environment {
    origin: Vector,
    sun: Vector,
    ambient_light: f32,
    diffuse_light: f32,
    grid_size: f64,
    surfaces: Vec<ColoredSurface>,
    spheres: Vec<Sphere>,
}

const IMAGE_SIZE: (u32, u32) = (500, 500);
const VOID_COLOR: [u8; 3] = [30, 30, 30];

// ========================== Ray casting ==========================================================

fn compute_lights(env: &Environment, surface: &Triangle, pt: Point) -> f32 {
    let sun_ray = Line {
        direction: vector!(pt, env.sun),
        origin: pt,
    };
    let covered = env
        .surfaces
        .iter()
        .filter(|sf| !sf.triangle.contains(pt))
        .map(|sf| sf.triangle.intersect(&sun_ray))
        // check if any intersection lies on the positive direction of the ray
        .any(|opt| opt.map(|t| t >= -FLOAT_EPS).unwrap_or(false));
    let different_halves = surface.plane.subs(env.origin) * surface.plane.subs(env.sun) <= 0.0;
    if covered || different_halves {
        env.ambient_light
    } else {
        let normal = surface.plane.normal();
        let cos = sun_ray.direction.cos(normal).abs() as f32;
        (1.0 - env.diffuse_light) + cos * env.diffuse_light
    }
}

enum Intersection<'a> {
    Triangle(f64, &'a ColoredSurface),
    Sphere(f64, &'a Sphere),
}
fn cast_ray(env: &Environment, ray: &Line) -> [u8; 3] {
    // Check intersections with triangles
    let triangle_intersection_opt = env
        .surfaces
        .iter()
        .map(|sf| sf.triangle.intersect(ray).map(|t| (t, sf)))
        .filter(Option::is_some)
        .map(Option::unwrap)
        .filter(|(t, _)| *t >= -FLOAT_EPS)
        .min_by(|(t1, _), (t2, _)| t1.partial_cmp(t2).unwrap());

    // Check intersections with spheres
    let sphere_intersection_opt = env
        .spheres
        .iter()
        .map(|sphere| sphere.intersect(ray).map(|t| (t, sphere)))
        .filter(Option::is_some)
        .map(Option::unwrap)
        .filter(|(t, _)| *t >= -FLOAT_EPS)
        .min_by(|(t1, _), (t2, _)| t1.partial_cmp(t2).unwrap());

    // Determine the closest intersection
    let closest_intersection = match (triangle_intersection_opt, sphere_intersection_opt) {
        (Some((t_tri, surface)), Some((t_sph, sphere))) => {
            if t_tri < t_sph {
                Some(Intersection::Triangle(t_tri, surface))
            } else {
                println!("Ray intersected with sphere at distance {}", t_sph);
                Some(Intersection::Sphere(t_sph, sphere))
            }
        }
        (None, Some((t, sphere))) => Some(Intersection::Sphere(t, sphere)),
        (Some((t, surface)), None) => Some(Intersection::Triangle(t, surface)),
        (None, None) => None,
    };

    // Handle the closest intersection
    match closest_intersection {
        Some(Intersection::Triangle(t, surface)) => {
            let brightness = compute_lights(&env, &surface.triangle, ray.at(t));
            surface
                .color
                .iter()
                .map(|c| (*c as f32 * brightness) as u8)
                .try_collect()
                .unwrap()
        }
        Some(Intersection::Sphere(t, sphere)) => {
            // Compute color and lighting for sphere intersection
            // You need to implement the logic for this part based on how you want spheres to be rendered
            // For example, you can return a constant color or compute shading based on the sphere's properties
            // This is a placeholder color
            [255, 0, 0]
        }
        None => VOID_COLOR,
    }
}

fn create_ray(env: &Environment, (x, y): (u32, u32)) -> Line {
    let interpolated = |cur: u32, max: u32| -> f64 { 2f64 * (cur as f64 / max as f64) - 1f64 };
    let vx = vector!(cross env.origin, vector!(axis y)).normalized();
    let vy = vector!(cross env.origin, vx).normalized();
    let pt = vector!()
        + interpolated(y, IMAGE_SIZE.1) * env.grid_size * vx
        + interpolated(x, IMAGE_SIZE.0) * env.grid_size * vy;
    Line {
        direction: vector!(env.origin, pt),
        origin: env.origin,
    }
}

fn cast_rays(env: &Environment, pool: &mut Pool) -> Vec<u8> {
    let pixel_count: usize = (IMAGE_SIZE.0 * IMAGE_SIZE.1) as usize;
    let chunks_size = pixel_count / num_cpus::get();
    let mut buff: Vec<[u8; 3]> = vec![[0, 0, 0]; pixel_count];
    pool.scoped(|scope| {
        let mut offset = 0;
        for chunk in buff.chunks_mut(chunks_size) {
            let chunk_len = chunk.len();
            scope.execute(move || {
                let rays = (0..chunk.len() as u32)
                    .map(|i| i + offset)
                    .map(|i| (i / IMAGE_SIZE.1, i % IMAGE_SIZE.1))
                    .map(|cords| create_ray(&env, cords));
                for (pixel, ray) in chunk.iter_mut().zip(rays) {
                    *pixel = cast_ray(&env, &ray);
                }
            });
            offset += chunk_len as u32;
        }
    });

    // dirty but fast cast from Vec<[u8;3]> to Vec<u8>
    unsafe {
        buff.set_len(buff.len() * 3);
        std::mem::transmute(buff)
    }
}

// ========================== Helper ===============================================================

fn parse_wavefront(filename: &str) -> (Vec<ColoredSurface>, Vec<Sphere>) {
    let mut out = vec![];
    let mut spheres: Vec<Sphere> = vec![];
    let mut min_y: f64 = 0.0;
    let mut max_dim: f64 = 0.0;
    let y_offset = 10.0;

    // First, parse the file for spheres
    let input = BufReader::new(File::open(filename).unwrap());
    for line in input.lines() {
        let line = line.unwrap();
        if line.starts_with("#sphere") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 5 {
                let x = parts[1].parse::<f64>().unwrap();
                let y = parts[2].parse::<f64>().unwrap();
                let z = parts[3].parse::<f64>().unwrap();
                let radius = parts[4].parse::<f64>().unwrap();
                spheres.push(Sphere::new(point!(x, y, z), radius));
                println!(
                    "Parsed sphere: center = ({}, {}, {}), radius = {}",
                    x, y, z, radius
                );
            }
        }
    }

    // Then, parse the file for triangles
    let input = BufReader::new(File::open(filename).unwrap());
    let model: Obj = load_obj(input).unwrap();
    let color = [255, 100, 100];

    for tri_indices_chunk in model.indices.chunks(3) {
        let vertices: [obj::Vertex; 3] = tri_indices_chunk
            .iter()
            .map(|idx| model.vertices[*idx as usize])
            .try_collect()
            .unwrap();

        let points: [Point; 3] = ArrayIntoIter::into_iter(vertices)
            .inspect(|vert| {
                vert.position
                    .iter()
                    .for_each(|coord| max_dim = max_dim.max(coord.abs() as f64))
            })
            .map(|vert| {
                point!(
                    vert.position[0],
                    vert.position[1] - y_offset,
                    vert.position[2]
                )
            })
            .inspect(|vert| min_y = min_y.min(vert.y))
            .try_collect()
            .unwrap();

        match triangle(points[0], points[1], points[2]) {
            Ok(triangle) => out.push(ColoredSurface { triangle, color }),
            Err(err) => eprintln!("{:?}", err),
        }
    }

    // push plane at minimum height
    let size = 60.0;
    let (tri1, tri2) = plane(point!(0, min_y - 2.0 * FLOAT_EPS, 0), size, size).unwrap();
    out.push(ColoredSurface {
        triangle: tri1,
        color: [200, 200, 200],
    });
    out.push(ColoredSurface {
        triangle: tri2,
        color: [200, 200, 200],
    });

    (out, spheres)
}

fn main() {
    let (surfaces, spheres) = parse_wavefront("test/tower.obj");
    let mut env = Environment {
        origin: vector!(-5, 70, 0),
        sun: vector!(-80, 150, 80),
        ambient_light: 0.4,
        diffuse_light: 0.2,
        grid_size: 40.0,
        surfaces, // Use parsed surfaces here
        spheres,  // Use parsed spheres here
    };
    let mut thread_pool = Pool::new(num_cpus::get() as u32);
    let origin_radius = 160f64;
    let steps: usize = 65;

    for step in 0..20 {
        let percent = step as f64 / steps as f64;
        let angle: f64 = percent * 2.0 * std::f64::consts::PI;
        env.origin.x = angle.sin() * origin_radius;
        env.origin.z = angle.cos() * origin_radius;

        let buffer = cast_rays(&env, &mut thread_pool);
        image::save_buffer(
            &Path::new(&format!("test/output{}.png", step)),
            &buffer,
            IMAGE_SIZE.0,
            IMAGE_SIZE.1,
            image::ColorType::Rgb8,
        )
        .expect("failed to write image");
    }
}
