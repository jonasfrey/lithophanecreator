//! Project files: every setting plus references to the picture and the outline STL.
//!
//! The referenced files are stored relative to the project file (so a folder holding the
//! project and its files can be moved or copied) and as absolute paths as a fallback.

use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::model::Settings;

pub const EXTENSION: &str = "lithoproj";
const FORMAT: &str = "lithophane-creator-project";
const VERSION: u32 = 1;

#[derive(Debug, Serialize, Deserialize)]
struct FileRef {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    relative: Option<PathBuf>,
    absolute: PathBuf,
}

#[derive(Debug, Serialize, Deserialize)]
struct ProjectFile {
    format: String,
    version: u32,
    #[serde(default)]
    image: Option<FileRef>,
    #[serde(default)]
    outline: Option<FileRef>,
    settings: Settings,
}

pub fn is_project_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case(EXTENSION))
}

pub fn save(path: &Path, settings: &Settings) -> Result<(), String> {
    let fail = |e: &dyn std::fmt::Display| format!("cannot save {}: {e}", path.display());
    let path = absolute(path).map_err(|e| fail(&e))?;
    let dir = path.parent().unwrap_or(Path::new("/"));
    let mut s = settings.clone();
    let file_ref = |p: PathBuf| -> Result<FileRef, String> {
        let absolute = absolute(&p).map_err(|e| fail(&e))?;
        Ok(FileRef {
            relative: relative_path(&absolute, dir),
            absolute,
        })
    };
    let file = ProjectFile {
        format: FORMAT.into(),
        version: VERSION,
        image: s.image_path.take().map(file_ref).transpose()?,
        outline: s.shape.stl_path.take().map(file_ref).transpose()?,
        settings: s,
    };
    let mut json = serde_json::to_string_pretty(&file).map_err(|e| fail(&e))?;
    json.push('\n');
    // Write next to the target and rename, so a failed write never truncates the project.
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    let tmp = path.with_file_name(format!(".{name}.tmp"));
    std::fs::write(&tmp, json)
        .and_then(|_| std::fs::rename(&tmp, &path))
        .map_err(|e| {
            std::fs::remove_file(&tmp).ok();
            fail(&e)
        })
}

/// Reads a project. Referenced files are looked up next to the project first, then at their
/// absolute location; if neither exists the absolute path is kept so the caller can report it.
pub fn load(path: &Path) -> Result<Settings, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let file: ProjectFile = serde_json::from_str(&text)
        .map_err(|e| format!("{} is not a valid project file: {e}", path.display()))?;
    if file.format != FORMAT {
        return Err(format!("{} is not a lithophane project", path.display()));
    }
    if file.version > VERSION {
        return Err(format!(
            "{} was saved by a newer version of this application",
            path.display()
        ));
    }
    let path = absolute(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let dir = path.parent().unwrap_or(Path::new("/"));
    let resolve = |r: FileRef| {
        r.relative
            .map(|rel| normalized(&dir.join(rel)))
            .filter(|p| p.exists())
            .unwrap_or(r.absolute)
    };
    let mut s = file.settings;
    s.image_path = file.image.map(resolve);
    s.shape.stl_path = file.outline.map(resolve);
    Ok(s)
}

fn absolute(p: &Path) -> std::io::Result<PathBuf> {
    std::path::absolute(p).map(|p| normalized(&p))
}

/// Removes `.` and resolves `..` lexically.
fn normalized(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    out.push("..");
                }
            }
            c => out.push(c),
        }
    }
    out
}

/// `target` relative to the directory `base`; both absolute. None across drives/roots.
fn relative_path(target: &Path, base: &Path) -> Option<PathBuf> {
    let t: Vec<Component> = target.components().collect();
    let b: Vec<Component> = base.components().collect();
    if t.first() != b.first() {
        return None;
    }
    let common = t.iter().zip(&b).take_while(|(x, y)| x == y).count();
    let mut r = PathBuf::new();
    for _ in common..b.len() {
        r.push("..");
    }
    for c in &t[common..] {
        r.push(c);
    }
    Some(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ShapeKind;

    fn temp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "lithophane-project-test-{name}-{}",
            std::process::id()
        ));
        std::fs::remove_dir_all(&d).ok();
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn relative_paths() {
        let rel = |t: &str, b: &str| relative_path(Path::new(t), Path::new(b));
        assert_eq!(rel("/a/b/c.png", "/a/b"), Some("c.png".into()));
        assert_eq!(rel("/a/img/c.png", "/a/proj"), Some("../img/c.png".into()));
        assert_eq!(rel("/x/c.png", "/a/b"), Some("../../x/c.png".into()));
        assert_eq!(normalized(Path::new("/a/./b/../c")), PathBuf::from("/a/c"));
    }

    #[test]
    fn roundtrip_and_move() {
        let root = temp_dir("roundtrip");
        let (proj_dir, img_dir) = (root.join("proj"), root.join("pictures"));
        std::fs::create_dir_all(&proj_dir).unwrap();
        std::fs::create_dir_all(&img_dir).unwrap();
        std::fs::write(img_dir.join("photo.png"), b"x").unwrap();
        std::fs::write(proj_dir.join("outline.stl"), b"x").unwrap();

        let mut s = Settings {
            image_path: Some(img_dir.join("photo.png")),
            resolution: 0.15,
            ..Settings::default()
        };
        s.shape.stl_path = Some(proj_dir.join("outline.stl"));
        s.shape.kind = ShapeKind::Stl;
        s.adjustments.gamma = 1.37;
        s.litho.offset = [0.123456789, -0.5];
        s.supports.foot_gap = 2.25;
        let file = proj_dir.join("test.lithoproj");
        save(&file, &s).unwrap();
        assert!(is_project_file(&file));
        let text = std::fs::read_to_string(&file).unwrap();
        assert!(text.contains("\"../pictures/photo.png\""), "{text}");
        assert!(text.contains("\"outline.stl\""), "{text}");
        assert_eq!(load(&file).unwrap(), s);

        // Moving the whole folder keeps the references working through the relative paths.
        let moved = root.join("moved");
        std::fs::create_dir_all(&moved).unwrap();
        std::fs::rename(&proj_dir, moved.join("proj")).unwrap();
        std::fs::rename(&img_dir, moved.join("pictures")).unwrap();
        let l = load(&moved.join("proj/test.lithoproj")).unwrap();
        assert_eq!(l.image_path, Some(moved.join("pictures/photo.png")));
        assert_eq!(l.shape.stl_path, Some(moved.join("proj/outline.stl")));

        // Missing files keep their absolute path for the error message.
        std::fs::remove_file(moved.join("pictures/photo.png")).unwrap();
        let l = load(&moved.join("proj/test.lithoproj")).unwrap();
        assert_eq!(l.image_path, Some(img_dir.join("photo.png")));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn rejects_foreign_files() {
        let root = temp_dir("foreign");
        let f = root.join("other.lithoproj");
        std::fs::write(
            &f,
            "{\"format\":\"something\",\"version\":1,\"settings\":{}}",
        )
        .unwrap();
        assert!(load(&f).unwrap_err().contains("not a lithophane project"));
        std::fs::write(&f, "not json").unwrap();
        assert!(load(&f).unwrap_err().contains("not a valid project file"));
        // Missing settings fields fall back to defaults (older or hand-written projects).
        std::fs::write(
            &f,
            format!(
                "{{\"format\":\"{FORMAT}\",\"version\":1,\"settings\":{{\"resolution\":0.3}}}}"
            ),
        )
        .unwrap();
        let s = load(&f).unwrap();
        assert_eq!(s.resolution, 0.3);
        assert_eq!(s.litho, Settings::default().litho);
        std::fs::remove_dir_all(&root).ok();
    }
}
