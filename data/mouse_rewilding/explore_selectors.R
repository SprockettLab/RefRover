library(tidyverse)
library(ggplot2)

SELECTOR_LABELS <- c(
  random          = "Random",
  top_containment = "Top containment",
  top_var         = "Top variance",
  maxmin          = "MaxMin diversity"
)
SELECTOR_COLORS <- c(
  random          = "#999999",
  top_containment = "#E69F00",
  top_var         = "#56B4E9",
  maxmin          = "#009E73"
)

dir.create("plots", showWarnings = FALSE)

# ── 1. Variance-explained curves ──────────────────────────────────────────────

comp <- read_tsv("selector_comparison.tsv", show_col_types = FALSE) |>
  mutate(selector = factor(selector, levels = names(SELECTOR_LABELS)))

comp_long <- comp |>
  pivot_longer(c(variance_explained, unique_var_explained),
               names_to = "metric", values_to = "value") |>
  mutate(metric = recode(metric,
    variance_explained  = "Sum of variances\n(overcounts correlated selections)",
    unique_var_explained = "Unique variance\n(orthogonal projection onto selected set)"
  ))

comp_summary <- comp_long |>
  group_by(selector, k, metric) |>
  summarise(
    mean_ve = mean(value),
    se_ve   = sd(value) / sqrt(n()),
    .groups = "drop"
  )

p_curves <- ggplot(comp_summary, aes(x = k, y = mean_ve, color = selector)) +
  geom_ribbon(aes(ymin = mean_ve - se_ve, ymax = mean_ve + se_ve, fill = selector),
              alpha = 0.12, color = NA) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 2.5) +
  facet_wrap(~metric, scales = "free_y") +
  scale_color_manual(values = SELECTOR_COLORS, labels = SELECTOR_LABELS) +
  scale_fill_manual(values  = SELECTOR_COLORS, labels = SELECTOR_LABELS) +
  scale_x_continuous(breaks = c(3, 5, 8, 10)) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(
    title    = "Variance explained by selector and k",
    subtitle = "Left: naive metric (overcounts); Right: orthogonal projection (correct)",
    x        = "k (number of prototypes per sample)",
    y        = "Variance explained (mean +/- SE)",
    color    = "Selector", fill = "Selector"
  ) +
  theme_bw(base_size = 11) +
  theme(legend.position = "right",
        strip.text = element_text(size = 9))

ggsave("plots/selector_variance_curves.pdf", p_curves, width = 10, height = 5)
ggsave("plots/selector_variance_curves.png", p_curves, width = 10, height = 5, dpi = 150)
cat("Saved plots/selector_variance_curves.pdf\n")

# ── 2. Per-sample distribution of variance explained ─────────────────────────

p_dist <- comp |>
  filter(k == 5) |>
  mutate(selector = fct_reorder(selector, unique_var_explained, .fun = median)) |>
  ggplot(aes(x = selector, y = unique_var_explained, fill = selector)) +
  geom_violin(trim = TRUE, scale = "width", alpha = 0.8) +
  geom_boxplot(width = 0.15, fill = "white", outlier.size = 0.8, outlier.alpha = 0.5) +
  scale_fill_manual(values = SELECTOR_COLORS, labels = SELECTOR_LABELS) +
  scale_x_discrete(labels = SELECTOR_LABELS) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(
    title    = "Per-sample unique variance explained at k=5",
    subtitle = "Orthogonal projection metric (removes correlated redundancy)",
    x        = NULL, y = "Unique variance explained"
  ) +
  theme_bw(base_size = 12) +
  theme(legend.position = "none", axis.text.x = element_text(angle = 20, hjust = 1))

ggsave("plots/selector_variance_dist.pdf", p_dist, width = 6, height = 5)
ggsave("plots/selector_variance_dist.png", p_dist, width = 6, height = 5, dpi = 150)
cat("Saved plots/selector_variance_dist.pdf\n")

# ── 3. Selector agreement heatmap at k=5 ─────────────────────────────────────

agree_raw <- read_tsv("selector_agreement_k5.tsv", show_col_types = FALSE)
agree <- agree_raw |>
  rename(s1 = 1) |>
  pivot_longer(-s1, names_to = "s2", values_to = "jaccard") |>
  mutate(
    s1 = factor(s1, levels = names(SELECTOR_LABELS), labels = SELECTOR_LABELS),
    s2 = factor(s2, levels = names(SELECTOR_LABELS), labels = SELECTOR_LABELS),
  )

p_agree <- ggplot(agree, aes(x = s2, y = s1, fill = jaccard)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.2f", jaccard)), size = 4) +
  scale_fill_gradient(low = "#f7fbff", high = "#08519c", name = "Mean\nJaccard",
                      limits = c(0, 1)) +
  labs(
    title    = "Selector agreement at k=5",
    subtitle = "Mean pairwise Jaccard overlap of selected prototype sets across all samples",
    x = NULL, y = NULL
  ) +
  coord_fixed() +
  theme_bw(base_size = 11) +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))

ggsave("plots/selector_agreement_heatmap.pdf", p_agree, width = 6, height = 5)
ggsave("plots/selector_agreement_heatmap.png", p_agree, width = 6, height = 5, dpi = 150)
cat("Saved plots/selector_agreement_heatmap.pdf\n")

# ── 4. Adaptive k distribution ────────────────────────────────────────────────

ak <- read_tsv("adaptive_k_ground_truth.tsv", show_col_types = FALSE)

cat("\nAdaptive k summary:\n")
print(ak |> count(optimal_k) |> arrange(optimal_k))

cat("\nAdaptive k by strain and timepoint:\n")
print(ak |> group_by(Mouse_Strain, Time_Point) |>
      summarise(median_k = median(optimal_k), mean_k = mean(optimal_k), .groups = "drop"))

p_ak <- ak |>
  filter(!is.na(Mouse_Strain)) |>
  ggplot(aes(x = factor(optimal_k), fill = Time_Point)) +
  geom_bar(position = "dodge") +
  facet_wrap(~Mouse_Strain) +
  scale_fill_manual(values = c(PRE = "#6baed6", POST = "#2ca25f")) +
  labs(
    title    = "Per-sample optimal k (from containment saturation curve)",
    subtitle = "Elbow of MaxMin residual-variance gain curve",
    x        = "Optimal k", y = "Number of samples",
    fill     = "Timepoint"
  ) +
  theme_bw(base_size = 11)

ggsave("plots/adaptive_k_distribution.pdf", p_ak, width = 7, height = 4)
ggsave("plots/adaptive_k_distribution.png", p_ak, width = 7, height = 4, dpi = 150)
cat("Saved plots/adaptive_k_distribution.pdf\n")

# ── 5. Saturation curves for representative samples ───────────────────────────

curves <- read_tsv("saturation_curves.tsv", show_col_types = FALSE)
manifest <- read_tsv("sample_manifest.tsv", show_col_types = FALSE)

# Pick 3 representative sample IDs (one per strain/timepoint combo)
rep_samples <- manifest |>
  group_by(Mouse_Strain, Time_Point) |>
  slice_head(n = 1) |>
  pull(Sample_ID) |>
  intersect(unique(curves$sample_id)) |>
  head(6)

p_sat <- curves |>
  filter(sample_id %in% rep_samples) |>
  left_join(select(manifest, sample_id = Sample_ID, Mouse_Strain, Time_Point), by = "sample_id") |>
  mutate(label = paste0(Mouse_Strain, " ", Time_Point, "\n(", sample_id, ")")) |>
  ggplot(aes(x = k)) +
  geom_col(aes(y = marginal_var), fill = "#6baed6", alpha = 0.7) +
  geom_line(aes(y = cumulative_var), color = "#08519c", linewidth = 1) +
  facet_wrap(~label, ncol = 3) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 0.1)) +
  labs(
    title    = "Saturation curves for representative samples",
    subtitle = "Bars = marginal residual variance; line = cumulative variance explained",
    x        = "k (prototype number, MaxMin order)",
    y        = "Fraction of total cross-sample variance"
  ) +
  theme_bw(base_size = 10)

ggsave("plots/saturation_curves.pdf", p_sat, width = 9, height = 5)
ggsave("plots/saturation_curves.png", p_sat, width = 9, height = 5, dpi = 150)
cat("Saved plots/saturation_curves.pdf\n")
