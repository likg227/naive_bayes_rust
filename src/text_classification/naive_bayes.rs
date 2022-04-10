use std::collections::hash_map::Keys;
use std::collections::HashMap;
use std::collections::HashSet;
use std::iter::FromIterator;
use std::vec::Vec;

struct Attributes {
    // Mapping from attribute to label to its frequency.
    attributes: HashMap<String, HashMap<String, i64>>,
}

impl Attributes {
    pub fn new() -> Attributes {
        Attributes {
            attributes: HashMap::new(),
        }
    }

    fn add(&mut self, attribute: &str, label: &str) {
        let labels = self
            .attributes
            .entry(attribute.to_string())
            .or_insert_with(HashMap::new);
        let value = labels.entry((*label).to_string()).or_insert(0);
        *value += 1;
    }

    fn get_frequency(&self, attribute: &str, label: &str) -> (Option<&i64>, bool) {
        match self.attributes.get(attribute) {
            Some(labels) => match labels.get(label) {
                Some(value) => (Some(value), true),
                None => (None, true),
            },
            None => (None, false),
        }
    }
}

struct Labels {
    counts: HashMap<String, i64>,
}

impl Labels {
    pub fn new() -> Labels {
        Labels {
            counts: HashMap::new(),
        }
    }

    fn add(&mut self, label: &str) {
        let value = self.counts.entry(label.to_string()).or_insert(0);
        *value += 1;
    }

    fn get_count(&self, label: &str) -> Option<&i64> {
        self.counts.get(label)
    }

    fn get_labels(&self) -> Keys<String, i64> {
        self.counts.keys()
    }

    fn get_total(&self) -> i64 {
        return self.counts.values().sum();
    }
}

struct Model {
    labels: Labels,
    attributes: Attributes,
}

impl Model {
    pub fn new() -> Model {
        Model {
            labels: Labels::new(),
            attributes: Attributes::new(),
        }
    }
    // Give each attribute (i.e. a piece of data) a label and count its frequency.
    fn train(&mut self, data: Vec<&str>, label: &str) {
        self.labels.add(label);
        for attribute in data {
            self.attributes.add(attribute, label);
        }
    }
}

pub struct NaiveBayes {
    model: Model,
    minimum_probability: f64,
}

impl Default for NaiveBayes {
    fn default() -> Self {
        Self::new()
    }
}

impl NaiveBayes {
    /// creates a new instance of a `NaiveBayes` classifier.
    pub fn new() -> NaiveBayes {
        NaiveBayes {
            model: Model::new(),
            minimum_probability: 1e-9,
        }
    }

    fn prior(&self, label: &str) -> Option<f64> {
        let total = self.model.labels.get_total() as f64;
        let label = &self.model.labels.get_count(label);
        if label.is_some() && total > 0.0 {
            Some(*label.unwrap() as f64 / total)
        } else {
            None
        }
    }

    fn calculate_attr_prob(&self, attribute: &str, label: &str) -> Option<f64> {
        match self.model.attributes.get_frequency(attribute, label) {
            (Some(frequency), true) => self
                .model
                .labels
                .get_count(label)
                .map(|count| (*frequency as f64) / (*count as f64)),
            (None, true) => Some(self.minimum_probability),
            (None, false) => None,
            (Some(_), false) => None,
        }
    }

    /// Given label, calculate the probability of attribute.
    fn label_prob(&self, label: &str, attrs: Vec<&str>) -> Vec<f64> {
        let mut probs: Vec<f64> = Vec::new();
        for attr in attrs {
            if let Some(p) = self.calculate_attr_prob(attr, label) {
                // println!("p: {}", p);
                probs.push(p);
            }
        }
        probs
    }

    pub fn train(&mut self, data: Vec<&str>, label: &str) {
        self.model.train(data, label);
    }

    pub fn classify(&self, data: Vec<&str>) -> String {
        let mut result: String = String::new();
        let mut p_max: f64 = 0.0;
        let labels: HashSet<String> =
            HashSet::from_iter(self.model.labels.get_labels().into_iter().cloned());
        for label in labels {
            let p = self.label_prob(&label, data.clone());
            let p_iter = p.into_iter().fold(1.0, |acc, x| acc * x);
            let prob = p_iter * self.prior(&label).unwrap();
            if prob >= p_max {
                p_max = prob;
                result = label;
            }
        }

        result
    }
}
