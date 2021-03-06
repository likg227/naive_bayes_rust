use std::{
    fs,
    sync::{Arc, Mutex},
    thread, vec,
};

mod text_classification;
use text_classification::naive_bayes::NaiveBayes;

use jieba_rs::Jieba;
use rand::prelude::*;

fn naive_bayes(labels: Vec<&'static str>, data_path: &'static str) {
    let nb = Arc::new(Mutex::new(Some(NaiveBayes::new())));
    let test_files = Arc::new(Mutex::new(Some(vec![])));
    let jieba = Arc::new(Jieba::new());

    // Train.
    let mut handles = vec![];
    for (index, label) in labels.into_iter().enumerate() {
        let nb = nb.clone();
        let test_files = test_files.clone();
        let path = <&str>::clone(&data_path);
        let jieba = jieba.clone();
        handles.push(thread::spawn(move || {
            let mut rng = thread_rng();
            let mut files = vec![];
            fs::read_dir(format!("{}{}", path, label))
                .unwrap()
                .for_each(|file| {
                    let content = fs::read_to_string(file.unwrap().path()).unwrap();
                    if rng.gen::<f64>() > 0.80 {
                        test_files
                            .lock()
                            .unwrap()
                            .as_mut()
                            .unwrap()
                            .push((index, content));
                    } else {
                        files.push(content);
                    }
                });
            for file in files {
                let words = jieba.cut(&file, false);
                nb.lock().unwrap().as_mut().unwrap().train(words, index);
            }
        }));
    }
    for handle in handles {
        handle.join().unwrap();
    }

    let nb = Arc::new(nb.lock().unwrap().take().unwrap());
    let test_files = test_files.lock().unwrap().take().unwrap();

    let total_count = test_files.len();
    let correct_count = Arc::new(Mutex::new(0));

    // Test.
    let mut handles = vec![];
    for (label, test_file) in test_files {
        let nb = nb.clone();
        let correct_count = correct_count.clone();
        let jieba = jieba.clone();
        handles.push(thread::spawn(move || {
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
        "????????????{}%",
        *correct_count.lock().unwrap() as f64 / total_count as f64 * 100_f64
    );
}

fn main() {
    // // ????????????
    // naive_bayes(
    //     vec!["car", "game", "it", "military"],
    //     "/Users/likaige/Code/course_design/naive_bayes_rust/data/topic_data/",
    // );
    // // ????????????
    // naive_bayes(
    //     vec!["pos", "neg"],
    //     "/Users/likaige/Code/course_design/naive_bayes_rust/data/emotion_data/",
    // );
    // // ????????????
    // naive_bayes(
    //     vec!["spam", "norm"],
    //     "/Users/likaige/Code/course_design/naive_bayes_rust/data/email_data/",
    // );
    // ????????????
    naive_bayes(
        vec!["spam", "norm"],
        "/Users/likaige/Code/course_design/naive_bayes_rust/data/short_message_data/",
    );
}
