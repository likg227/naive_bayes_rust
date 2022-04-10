use std::{
    fs,
    sync::{Arc, Mutex},
    thread,
};

mod text_classification;
use text_classification::naive_bayes::NaiveBayes;

use jieba_rs::Jieba;
use rand::prelude::*;

fn main() {
    let nb = Arc::new(Mutex::new(Some(NaiveBayes::new())));
    let labels: Vec<&str> = vec!["car", "game", "it", "military"];
    let mut handles = vec![];
    let test_files = Arc::new(Mutex::new(Some(vec![])));

    for label in labels {
        let nb = nb.clone();
        let test_files = test_files.clone();
        handles.push(thread::spawn(move || {
            let mut rng = thread_rng();
            let mut files = vec![];
            fs::read_dir(format!(
                "/home/stephen/Code/naive_bayes_rust/data/{}",
                label
            ))
            .unwrap()
            .for_each(|file| {
                let content = fs::read_to_string(file.unwrap().path()).unwrap();
                if rng.gen::<f64>() > 0.85 {
                    test_files
                        .lock()
                        .unwrap()
                        .as_mut()
                        .unwrap()
                        .push((label, content));
                } else {
                    files.push(content);
                }
            });
            let jieba = Jieba::new();
            for file in files {
                let words = jieba.cut(&file, false);
                nb.lock().unwrap().as_mut().unwrap().train(words, label);
            }
        }));
    }
    for handle in handles {
        handle.join().unwrap();
    }

    let nb = Arc::new(nb.lock().unwrap().take().unwrap());
    let mut handles = vec![];
    let test_files = test_files.lock().unwrap().take().unwrap();
    let total_count = test_files.len();
    let correct_count = Arc::new(Mutex::new(0));
    for (label, test_file) in test_files {
        let nb = nb.clone();
        let correct_count = correct_count.clone();
        handles.push(thread::spawn(move || {
            let jieba = Jieba::new();
            let words = jieba.cut(&test_file, false);
            let actual_label = nb.classify(words);
            if label == actual_label {
                *correct_count.lock().unwrap() += 1;
            }
        }));
    }
    for handle in handles {
        handle.join().unwrap();
    }

    println!(
        "正确率：{}%",
        *correct_count.lock().unwrap() as f64 / total_count as f64 * 100_f64
    );
}