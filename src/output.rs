//! Turning the model's token stream into pixel-space boxes / points.

use image::{Rgb, RgbImage};
use regex::Regex;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Detection {
    Box { label: Option<String>, x1: f32, y1: f32, x2: f32, y2: f32 },
    Point { label: Option<String>, x: f32, y: f32 },
}

/// Parses `<ref>label</ref><box><x1><y1><x2><y2></box>...` (coordinates are
/// integers in [0, 1000]) into pixel coordinates of a `width`x`height` image.
pub fn parse(answer: &str, width: u32, height: u32) -> Vec<Detection> {
    let re = Regex::new(r"<ref>(.*?)</ref>|<box>((?:<\d+>)+)</box>").unwrap();
    let num = Regex::new(r"<(\d+)>").unwrap();
    let (w, h) = (width as f32 / 1000.0, height as f32 / 1000.0);
    let mut label: Option<String> = None;
    let mut out = Vec::new();
    for cap in re.captures_iter(answer) {
        if let Some(l) = cap.get(1) {
            label = Some(l.as_str().trim().to_string());
            continue;
        }
        let c: Vec<f32> = num.captures_iter(&cap[2]).filter_map(|m| m[1].parse().ok()).collect();
        match c[..] {
            [x1, y1, x2, y2] => out.push(Detection::Box {
                label: label.clone(),
                x1: x1 * w,
                y1: y1 * h,
                x2: x2 * w,
                y2: y2 * h,
            }),
            [x, y] => out.push(Detection::Point { label: label.clone(), x: x * w, y: y * h }),
            _ => {}
        }
    }
    out
}

const PALETTE: [[u8; 3]; 8] = [
    [230, 25, 75],
    [60, 180, 75],
    [0, 130, 200],
    [245, 130, 48],
    [145, 30, 180],
    [70, 240, 240],
    [240, 50, 230],
    [210, 245, 60],
];

/// Draws detections onto `img` (one colour per label).
pub fn draw(img: &mut RgbImage, dets: &[Detection]) {
    let mut labels: Vec<Option<String>> = Vec::new();
    let mut color_for = |l: &Option<String>| {
        let i = labels.iter().position(|x| x == l).unwrap_or_else(|| {
            labels.push(l.clone());
            labels.len() - 1
        });
        Rgb(PALETTE[i % PALETTE.len()])
    };
    let thick = ((img.width().max(img.height()) / 400) as i64).max(2);
    for d in dets {
        match d {
            Detection::Box { label, x1, y1, x2, y2 } => {
                let c = color_for(label);
                for t in 0..thick {
                    rect(img, *x1 as i64 + t, *y1 as i64 + t, *x2 as i64 - t, *y2 as i64 - t, c);
                }
            }
            Detection::Point { label, x, y } => {
                let c = color_for(label);
                let r = thick * 3;
                for dy in -r..=r {
                    for dx in -r..=r {
                        if dx * dx + dy * dy <= r * r {
                            put(img, *x as i64 + dx, *y as i64 + dy, c);
                        }
                    }
                }
            }
        }
    }
}

fn put(img: &mut RgbImage, x: i64, y: i64, c: Rgb<u8>) {
    if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
        img.put_pixel(x as u32, y as u32, c);
    }
}

fn rect(img: &mut RgbImage, x1: i64, y1: i64, x2: i64, y2: i64, c: Rgb<u8>) {
    for x in x1..=x2 {
        put(img, x, y1, c);
        put(img, x, y2, c);
    }
    for y in y1..=y2 {
        put(img, x1, y, c);
        put(img, x2, y, c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_refs_boxes_points() {
        let a = "<ref>person</ref><box><100><200><300><400></box><box><0><0><1000><1000></box><ref>cup</ref><box><500><500></box><|im_end|>";
        let d = parse(a, 2000, 1000);
        assert_eq!(d.len(), 3);
        match &d[0] {
            Detection::Box { label, x1, y1, x2, y2 } => {
                assert_eq!(label.as_deref(), Some("person"));
                assert_eq!((*x1, *y1, *x2, *y2), (200.0, 200.0, 600.0, 400.0));
            }
            _ => panic!(),
        }
        assert!(matches!(&d[2], Detection::Point { label: Some(l), x, y } if l == "cup" && *x == 1000.0 && *y == 500.0));
    }
}
