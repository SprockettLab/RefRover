#!/usr/bin/env Rscript
# Analyze RefRover benchmark results.
#
# Reads benchmark_results.tsv (from refrover aggregate-results) and answers:
#   1. Does selection rule matter?         heatmap + boxplots by rule
#   2. Does k matter?                      yield curve vs k per rule
#   3. Do proxy scores predict MAG yield?  scatter vs primary yield metric + Spearman r
#   4. Jaccard vs containment matrix?      cross-matrix agreement scatter
#   5. Which focal assemblies are hardest? per-focal variability boxplots
#
# Primary yield metric: sum_qs (Σ max(0, completeness − 5×contamination), Olm et al.
# 2017 ISME J) when present; falls back to weighted_mags (2×n_high + 1×n_medium).
#
# Usage:
#   Rscript scripts/analyze_benchmark.R \
#       --results work/benchmark/benchmark_results.tsv \
#       --outdir  work/analysis/
#
# Dependencies: tidyverse, patchwork
#   conda install -n refrover-benchmark -c conda-forge r-base r-tidyverse r-patchwork

suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
})

# ── argument parsing ──────────────────────────────────────────────────────────

args <- commandArgs(trailingOnly = TRUE)

parse_args <- function(args) {
  out <- list(results = NULL, outdir = "analysis")
  i <- 1
  while (i <= length(args)) {
    if      (args[i] == "--results") { out$results <- args[i + 1]; i <- i + 2 }
    else if (args[i] == "--outdir")  { out$outdir  <- args[i + 1]; i <- i + 2 }
    else { cat("Unknown argument:", args[i], "\n"); i <- i + 1 }
  }
  out
}

opt <- parse_args(args)
if (is.null(opt$results)) {
  cat("Usage: Rscript analyze_benchmark.R --results <tsv> [--outdir <dir>]\n")
  quit(status = 1)
}

dir.create(opt$outdir, recursive = TRUE, showWarnings = FALSE)

# ── colour palette ────────────────────────────────────────────────────────────

rule_colors <- c(
  random     = "#999999",
  maxmin     = "#4477AA",
  css        = "#EE6677",
  kmedoids   = "#228833",
  archetype  = "#CCBB44",
  greedy_var = "#AA3377"
)

# ── load and clean data ───────────────────────────────────────────────────────

cat("Loading", opt$results, "\n")
raw <- read_tsv(opt$results, show_col_types = FALSE)

# Separate successful cells from errors.
if ("error" %in% names(raw)) {
  df  <- filter(raw, is.na(error))
  n_err <- sum(!is.na(raw$error))
  cat(sprintf("Loaded %d cells: %d succeeded, %d errors\n",
              nrow(raw), nrow(df), n_err))
  if (n_err > 0)
    cat("  Error cells excluded — check error.json files for details\n")
} else {
  df <- raw
  cat(sprintf("Loaded %d cells\n", nrow(df)))
}

df <- df |>
  mutate(
    rule   = factor(rule),
    matrix = factor(matrix),
    k      = as.integer(k)
  )

# Determine which yield metric to use as the primary outcome.
# sum_qs  — continuous Quality Score (Olm et al. 2017): preferred, more defensible.
# weighted_mags — binary tier score (2×n_high + 1×n_medium): legacy fallback.
# If both present, plots show sum_qs as primary; weighted_mags available for comparison.
if ("sum_qs" %in% names(df)) {
  yvar  <- "sum_qs"
  ylabel <- "Quality Score sum  (Σ max(0, completeness − 5×contamination))"
  cat("Primary yield metric: sum_qs  (continuous, preferred)\n")
} else {
  yvar  <- "weighted_mags"
  ylabel <- "Weighted MAGs  (2×n_high + 1×n_medium)"
  cat("Primary yield metric: weighted_mags  (sum_qs not found in data)\n")
}

# ── helper: Spearman label ────────────────────────────────────────────────────

spearman_label <- function(x, y) {
  ok <- is.finite(x) & is.finite(y)
  if (sum(ok) < 5) return("r = n/a")
  t     <- cor.test(x[ok], y[ok], method = "spearman", exact = FALSE)
  p_str <- if (t$p.value < 0.001) "p<0.001" else sprintf("p=%.3f", t$p.value)
  sprintf("r = %.2f, %s", t$estimate, p_str)
}

# ── plot 1: heatmap ───────────────────────────────────────────────────────────

cat("Plot 1: heatmap\n")
heat_data <- df |>
  group_by(matrix, rule, k) |>
  summarise(mean_yield = mean(.data[[yvar]], na.rm = TRUE), .groups = "drop")

p1 <- ggplot(heat_data, aes(x = factor(k), y = rule, fill = mean_yield)) +
  geom_tile(colour = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.1f", mean_yield)), size = 3.2) +
  scale_fill_distiller(palette = "YlOrRd", direction = 1,
                       name = sprintf("mean\n%s", yvar)) +
  facet_wrap(~matrix, labeller = label_both) +
  labs(title = sprintf("Mean %s per (matrix, rule, k)", yvar),
       x = "k  (references per focal assembly)", y = NULL) +
  theme_minimal(base_size = 11) +
  theme(panel.grid = element_blank())

ggsave(file.path(opt$outdir, "01_heatmap.png"), p1,
       width = 5 * n_distinct(df$matrix), height = 4, dpi = 150)

# ── plot 2: k curves ──────────────────────────────────────────────────────────

cat("Plot 2: k curves\n")
p2 <- df |>
  group_by(matrix, rule, k) |>
  summarise(
    mean = mean(.data[[yvar]], na.rm = TRUE),
    sem  = sd(.data[[yvar]],   na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  ) |>
  ggplot(aes(x = k, y = mean, colour = rule, fill = rule)) +
  geom_ribbon(aes(ymin = mean - sem, ymax = mean + sem),
              alpha = 0.15, colour = NA) +
  geom_line(linewidth = 1) +
  geom_point(size = 2.5) +
  scale_colour_manual(values = rule_colors, aesthetics = c("colour", "fill")) +
  facet_wrap(~matrix, labeller = label_both) +
  labs(title = sprintf("%s vs k  (mean ± SEM over focal assemblies)", yvar),
       x = "k  (references per focal assembly)",
       y = ylabel,
       colour = "rule", fill = "rule") +
  theme_bw(base_size = 11)

ggsave(file.path(opt$outdir, "02_k_curves.png"), p2,
       width = 5 * n_distinct(df$matrix), height = 4, dpi = 150)

# ── plot 3: proxy score correlations ─────────────────────────────────────────

proxy_cols <- intersect(
  c("frac_variance", "effective_rank", "tiered_axis_count"),
  names(df)
)

if (length(proxy_cols) == 0) {
  cat("  No proxy score columns found — skipping plot 3\n")
} else {
  cat("Plot 3: proxy correlations\n")

  proxy_long <- df |>
    select(matrix, rule, k, focal, all_of(c(yvar, proxy_cols))) |>
    pivot_longer(all_of(proxy_cols), names_to = "proxy", values_to = "score")

  spearman_labels <- proxy_long |>
    group_by(matrix, proxy) |>
    summarise(
      label = spearman_label(score, .data[[yvar]]),
      .groups = "drop"
    ) |>
    mutate(x = -Inf, y = Inf, hjust = -0.1, vjust = 1.4)

  p3 <- ggplot(proxy_long,
               aes(x = score, y = .data[[yvar]], colour = rule)) +
    geom_point(alpha = 0.35, size = 1.2) +
    geom_smooth(method = "lm", se = FALSE, aes(group = 1),
                colour = "black", linewidth = 0.7, linetype = "dashed") +
    geom_text(data = spearman_labels,
              aes(x = x, y = y, label = label, hjust = hjust, vjust = vjust),
              colour = "black", size = 2.8, inherit.aes = FALSE) +
    scale_colour_manual(values = rule_colors) +
    facet_grid(proxy ~ matrix,
               scales = "free_x",
               labeller = labeller(matrix = label_both, proxy = label_value)) +
    labs(title = "Proxy scores vs real MAG yield",
         x = "proxy score value", y = ylabel,
         colour = "rule") +
    theme_bw(base_size = 10)

  ggsave(file.path(opt$outdir, "03_proxy_correlations.png"), p3,
         width = 4.5 * n_distinct(df$matrix),
         height = 3.5 * length(proxy_cols), dpi = 150)
}

# ── plot 4: Jaccard vs containment matrix agreement ──────────────────────────

if (n_distinct(df$matrix) < 2) {
  cat("  Only one matrix present — skipping plot 4\n")
} else {
  cat("Plot 4: matrix comparison\n")

  mat_wide <- df |>
    select(rule, k, focal, matrix, all_of(yvar)) |>
    pivot_wider(names_from = matrix, values_from = all_of(yvar)) |>
    drop_na()

  if (!all(c("jaccard", "containment") %in% names(mat_wide))) {
    cat("  Expected matrix names 'jaccard' and 'containment' — skipping plot 4\n")
  } else {
    spearman_mat <- mat_wide |>
      group_by(rule) |>
      summarise(
        label = spearman_label(jaccard, containment),
        .groups = "drop"
      ) |>
      mutate(x = -Inf, y = Inf, hjust = -0.1, vjust = 1.4)

    p4 <- ggplot(mat_wide, aes(x = jaccard, y = containment, colour = rule)) +
      geom_abline(slope = 1, intercept = 0,
                  linetype = "dashed", colour = "grey50", linewidth = 0.7) +
      geom_point(alpha = 0.45, size = 1.8) +
      geom_text(data = spearman_mat,
                aes(x = x, y = y, label = label, hjust = hjust, vjust = vjust),
                colour = "black", size = 3, inherit.aes = FALSE) +
      scale_colour_manual(values = rule_colors) +
      facet_wrap(~rule) +
      labs(title = sprintf("%s: Jaccard vs Containment feature matrix", yvar),
           subtitle = "Each point = one focal assembly × k cell",
           x = sprintf("Jaccard matrix  (%s)", yvar),
           y = sprintf("Containment matrix  (%s)", yvar),
           colour = "rule") +
      theme_bw(base_size = 11)

    ggsave(file.path(opt$outdir, "04_matrix_comparison.png"), p4,
           width = 4 * n_distinct(mat_wide$rule), height = 4.5, dpi = 150)
  }
}

# ── plot 5: per-focal variability ─────────────────────────────────────────────

cat("Plot 5: focal variability\n")
focal_order <- df |>
  group_by(focal) |>
  summarise(med = median(.data[[yvar]], na.rm = TRUE), .groups = "drop") |>
  arrange(desc(med)) |>
  pull(focal)

p5 <- df |>
  mutate(focal = factor(focal, levels = focal_order)) |>
  ggplot(aes(x = focal, y = .data[[yvar]])) +
  geom_boxplot(fill = "#4477AA", alpha = 0.55, outlier.size = 0.8,
               linewidth = 0.4) +
  labs(title = "Per-focal assembly variability  (across all rules × k × matrix)",
       x = "focal assembly  (ordered by median yield)",
       y = ylabel) +
  theme_bw(base_size = 10) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 6))

ggsave(file.path(opt$outdir, "05_focal_variability.png"), p5,
       width = max(8, n_distinct(df$focal) * 0.35), height = 4.5, dpi = 150)

# ── text summary ──────────────────────────────────────────────────────────────

cat(sprintf("\n── Mean %s by (matrix, rule, k) ──────────────────────────\n", yvar))
df |>
  group_by(matrix, rule, k) |>
  summarise(
    mean = round(mean(.data[[yvar]], na.rm = TRUE), 2),
    sd   = round(sd(.data[[yvar]],   na.rm = TRUE), 2),
    n    = n(),
    .groups = "drop"
  ) |>
  arrange(desc(mean)) |>
  print(n = 30)

if (length(proxy_cols) > 0) {
  cat(sprintf("\n── Spearman correlation: proxy score vs %s ──────────\n", yvar))
  df |>
    pivot_longer(all_of(proxy_cols), names_to = "proxy", values_to = "score") |>
    group_by(matrix, proxy) |>
    summarise(
      spearman = spearman_label(score, .data[[yvar]]),
      .groups = "drop"
    ) |>
    print()
}

cat(sprintf("\n── Top 10 (matrix, rule, k) combinations by mean %s ───\n", yvar))
df |>
  group_by(matrix, rule, k) |>
  summarise(
    mean_yield = round(mean(.data[[yvar]], na.rm = TRUE), 2),
    .groups = "drop"
  ) |>
  slice_max(mean_yield, n = 10) |>
  print()

cat(sprintf("\nDone. Plots saved to %s/\n", opt$outdir))
