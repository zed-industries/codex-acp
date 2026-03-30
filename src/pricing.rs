/// Estimated per-token pricing in USD for OpenAI models.
///
/// These rates are used to derive an approximate `UsageUpdate.cost` from token
/// counts when the upstream Codex event stream does not provide an authoritative
/// cost signal.  Prices will need periodic updates as OpenAI changes its
/// pricing.  Once <https://github.com/openai/codex/issues/16258> is resolved,
/// this module can be replaced with the upstream cost.

/// Per-million-token prices in USD.
pub(crate) struct ModelPricing {
    /// Regular (non-cached) input tokens.
    input: f64,
    /// Cached input tokens.
    cached_input: f64,
    /// Output tokens (includes reasoning tokens).
    output: f64,
}

impl ModelPricing {
    const fn new(input: f64, cached_input: f64, output: f64) -> Self {
        Self {
            input,
            cached_input,
            output,
        }
    }

    /// Estimate cost in USD from per-turn token counts.
    ///
    /// `input_tokens` is the total input count reported by the API (which
    /// *includes* cached tokens).  We subtract `cached_input_tokens` to get the
    /// non-cached portion that is billed at the regular input rate.
    pub(crate) fn estimate_cost(
        &self,
        input_tokens: u64,
        cached_input_tokens: u64,
        output_tokens: u64,
    ) -> f64 {
        let uncached = input_tokens.saturating_sub(cached_input_tokens) as f64;
        let cached = cached_input_tokens as f64;
        let output = output_tokens as f64;

        (uncached * self.input + cached * self.cached_input + output * self.output) / 1_000_000.0
    }
}

// Prices per 1 M tokens (USD) — last updated 2025-04.
// Sorted longest-prefix-first so the first match wins.
static PRICING_TABLE: &[(&str, ModelPricing)] = &[
    ("gpt-4.1-mini", ModelPricing::new(0.40, 0.10, 1.60)),
    ("gpt-4.1-nano", ModelPricing::new(0.10, 0.025, 0.40)),
    ("gpt-4.1", ModelPricing::new(2.00, 0.50, 8.00)),
    ("gpt-4o-mini", ModelPricing::new(0.15, 0.075, 0.60)),
    ("gpt-4o", ModelPricing::new(2.50, 1.25, 10.00)),
    ("o4-mini", ModelPricing::new(1.10, 0.275, 4.40)),
    ("o3-mini", ModelPricing::new(1.10, 0.55, 4.40)),
    ("o3", ModelPricing::new(2.00, 0.50, 8.00)),
];

/// Look up pricing for `model`.  Tries the full slug first, then strips a
/// trailing date suffix (e.g. `-2025-04-14`) and retries, matching against
/// known prefixes.
pub(crate) fn lookup_pricing(model: &str) -> Option<&'static ModelPricing> {
    // Try prefix match against the table (longest prefix listed first).
    if let Some(pricing) = prefix_match(model) {
        return Some(pricing);
    }

    // Strip a trailing `-YYYY-MM-DD` date suffix and retry.
    if let Some(base) = model.rsplit_once('-').and_then(|(left, _)| {
        // Only strip if the part before the last dash also ends with digits
        // (i.e. the suffix looks like a date: YYYY-MM-DD).
        left.rsplit_once('-')
            .and_then(|(ll, _)| ll.rsplit_once('-').map(|(lll, _)| lll))
    }) {
        return prefix_match(base);
    }

    None
}

fn prefix_match(slug: &str) -> Option<&'static ModelPricing> {
    PRICING_TABLE
        .iter()
        .find(|(prefix, _)| slug.starts_with(prefix))
        .map(|(_, p)| p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match() {
        let p = lookup_pricing("gpt-4.1").unwrap();
        assert!((p.input - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn date_suffix() {
        let p = lookup_pricing("gpt-4.1-2025-04-14").unwrap();
        assert!((p.input - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn mini_before_base() {
        let p = lookup_pricing("gpt-4.1-mini").unwrap();
        assert!((p.input - 0.40).abs() < f64::EPSILON);
    }

    #[test]
    fn nano_match() {
        let p = lookup_pricing("gpt-4.1-nano").unwrap();
        assert!((p.input - 0.10).abs() < f64::EPSILON);
    }

    #[test]
    fn o3_match() {
        let p = lookup_pricing("o3").unwrap();
        assert!((p.input - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn o4_mini_match() {
        let p = lookup_pricing("o4-mini").unwrap();
        assert!((p.input - 1.10).abs() < f64::EPSILON);
    }

    #[test]
    fn unknown_model() {
        assert!(lookup_pricing("unknown-model-xyz").is_none());
    }

    #[test]
    fn cost_calculation() {
        let p = lookup_pricing("gpt-4.1").unwrap();
        // 1000 uncached input (2000 total - 1000 cached), 1000 cached, 500 output
        // = (1000 * 2.0 + 1000 * 0.5 + 500 * 8.0) / 1_000_000
        // = (2000 + 500 + 4000) / 1_000_000
        // = 0.0065
        let cost = p.estimate_cost(2000, 1000, 500);
        assert!((cost - 0.0065).abs() < 1e-10);
    }
}
