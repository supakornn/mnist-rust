use ab_glyph::{FontRef, PxScale};
use flate2::read::GzDecoder;
use image::RgbImage;
use imageproc::drawing::draw_text_mut;
use std::fs;
use std::io::Read;

const COLS: u32 = 10;
const ROWS: u32 = 2;
const NUM_SAMPLES: usize = (COLS * ROWS) as usize;
const DIGIT_SIZE: u32 = 28;
const CELL_W: u32 = DIGIT_SIZE + 4;
const CELL_H: u32 = DIGIT_SIZE + 16;

fn load_images(path: &str) -> Vec<Vec<u8>> {
    let file = fs::File::open(path).expect("Cannot open image file");
    let mut decoder = GzDecoder::new(file);
    let mut buf = Vec::new();
    decoder.read_to_end(&mut buf).unwrap();

    let num_images = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    let pixel_count = 28 * 28;
    (0..num_images)
        .map(|i| buf[16 + i * pixel_count..16 + (i + 1) * pixel_count].to_vec())
        .collect()
}

fn load_labels(path: &str) -> Vec<u8> {
    let file = fs::File::open(path).expect("Cannot open label file");
    let mut decoder = GzDecoder::new(file);
    let mut buf = Vec::new();
    decoder.read_to_end(&mut buf).unwrap();

    let num_labels = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    buf[8..8 + num_labels].to_vec()
}

fn main() {
    let images = load_images("data/train-images-idx3-ubyte.gz");
    let labels = load_labels("data/train-labels-idx1-ubyte.gz");
    println!("Loaded {} images", images.len());

    let font_data = fs::read("/System/Library/Fonts/Helvetica.ttc").expect("Cannot load font");
    let font = FontRef::try_from_slice(&font_data).expect("Cannot parse font");
    let scale = PxScale::from(10.0);

    let canvas_w = COLS * CELL_W;
    let canvas_h = ROWS * CELL_H;
    let mut canvas = RgbImage::from_pixel(canvas_w, canvas_h, image::Rgb([255u8, 255, 255]));

    for (idx, (pixels, &label)) in images
        .iter()
        .zip(labels.iter())
        .take(NUM_SAMPLES)
        .enumerate()
    {
        let col = (idx as u32) % COLS;
        let row = (idx as u32) / COLS;
        let ox = col * CELL_W + 2;
        let oy = row * CELL_H + 2;

        for py in 0..DIGIT_SIZE {
            for px in 0..DIGIT_SIZE {
                let v = pixels[(py * DIGIT_SIZE + px) as usize];
                let pixel = image::Rgb([v, v, v]);
                canvas.put_pixel(ox + px, oy + py, pixel);
            }
        }

        let label_text = format!("Label: {}", label);
        draw_text_mut(
            &mut canvas,
            image::Rgb([0u8, 0, 0]),
            (ox) as i32,
            (oy + DIGIT_SIZE + 2) as i32,
            scale,
            &font,
            &label_text,
        );
    }

    fs::create_dir_all("images").unwrap();
    canvas
        .save("images/mnist_samples.png")
        .expect("Cannot save image");
    println!("Saved: images/mnist_samples.png");
}
