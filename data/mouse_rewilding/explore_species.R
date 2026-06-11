library(tidyverse)
library(ggplot2)

# ── Load data ──────────────────────────────────────────────────────────────────

species_mat <- read_tsv("gtdb_species_matrix.tsv") |>
  column_to_rownames("name")

manifest <- read_tsv("sample_manifest.tsv")

cat("Species matrix:", nrow(species_mat), "species x", ncol(species_mat), "samples\n")
cat("Non-zero entries:", sum(species_mat > 0), "\n")
cat("Sparsity:", round(mean(species_mat == 0) * 100, 1), "%\n\n")

# ── Per-species summary stats ──────────────────────────────────────────────────

species_stats <- tibble(
  name = rownames(species_mat),
  mean_abund  = rowMeans(species_mat),
  sd_abund    = apply(species_mat, 1, sd),
  prevalence  = rowSums(species_mat > 0),   # number of samples detected in
  max_abund   = apply(species_mat, 1, max),
) |>
  mutate(
    cv = sd_abund / (mean_abund + 1e-9),    # coefficient of variation
    log10_mean = log10(mean_abund + 1e-9),
  ) |>
  arrange(desc(mean_abund))

cat("Top 20 species by mean abundance:\n")
print(head(select(species_stats, name, mean_abund, cv, prevalence), 20))

# ── CV vs mean scatter (the key diagnostic) ──────────────────────────────────
# Three-regime hypothesis:
#   Low abundance / high CV  -> rare / intermittent taxa (noisy, little differential signal)
#   Mid abundance / high CV  -> core taxa varying between treatments (HIGH VALUE for selection)
#   High abundance / low CV  -> stable core (always present, constant depth)

# Restrict to species detected in at least 5 samples (filter extreme sparsity)
detected <- filter(species_stats, prevalence >= 5)
cat("\nSpecies in >=5 samples:", nrow(detected), "\n")

p_cv_mean <- ggplot(detected, aes(x = log10_mean, y = cv)) +
  geom_point(aes(color = prevalence, size = prevalence), alpha = 0.6) +
  scale_color_viridis_c(name = "Prevalence\n(n samples)", option = "plasma") +
  scale_size_continuous(name = "Prevalence", range = c(0.5, 3)) +
  geom_smooth(method = "loess", se = TRUE, color = "black", linewidth = 0.8) +
  labs(
    title = "Species-level abundance: CV vs mean",
    subtitle = paste0(nrow(detected), " species detected in >=5 samples"),
    x = "log10(mean f_unique_to_query)",
    y = "Coefficient of variation"
  ) +
  theme_bw(base_size = 11)

ggsave("plots/species_cv_vs_mean.pdf", p_cv_mean, width = 8, height = 6)
ggsave("plots/species_cv_vs_mean.png", p_cv_mean, width = 8, height = 6, dpi = 150)
cat("Saved plots/species_cv_vs_mean.pdf\n")

# ── SD vs mean (log-log) ──────────────────────────────────────────────────────
# At assembly level, SD scaled linearly with mean. Does this persist at species level?

p_sd_mean <- ggplot(detected, aes(x = log10_mean, y = log10(sd_abund + 1e-9))) +
  geom_point(aes(color = prevalence), alpha = 0.5, size = 0.8) +
  scale_color_viridis_c(option = "plasma") +
  geom_smooth(method = "lm", se = TRUE, color = "firebrick") +
  labs(
    title = "SD vs mean (log-log) — species level",
    subtitle = "Linear relationship indicates proportional variance; deviation = differential signal",
    x = "log10(mean)", y = "log10(SD)"
  ) +
  theme_bw(base_size = 11)

ggsave("plots/species_sd_vs_mean.pdf", p_sd_mean, width = 7, height = 5)
cat("Saved plots/species_sd_vs_mean.pdf\n")

# ── Strain comparison: C57 vs NYOB ───────────────────────────────────────────

c57_ids  <- manifest |> filter(Mouse_Strain == "C57")  |> pull(Sample_ID)
nyob_ids <- manifest |> filter(Mouse_Strain == "NYOB") |> pull(Sample_ID)

c57_mat  <- species_mat[, intersect(colnames(species_mat), c57_ids),  drop = FALSE]
nyob_mat <- species_mat[, intersect(colnames(species_mat), nyob_ids), drop = FALSE]

strain_stats <- tibble(
  name           = rownames(species_mat),
  mean_c57       = rowMeans(c57_mat),
  mean_nyob      = rowMeans(nyob_mat),
  cv_c57         = apply(c57_mat,  1, sd) / (rowMeans(c57_mat)  + 1e-9),
  cv_nyob        = apply(nyob_mat, 1, sd) / (rowMeans(nyob_mat) + 1e-9),
  prev_c57       = rowSums(c57_mat  > 0),
  prev_nyob      = rowSums(nyob_mat > 0),
) |>
  mutate(
    log10_mean_c57  = log10(mean_c57  + 1e-9),
    log10_mean_nyob = log10(mean_nyob + 1e-9),
  )

# Scatter: C57 mean vs NYOB mean (species consistently present in both vs strain-specific)
p_strain <- strain_stats |>
  filter(prev_c57 >= 3 | prev_nyob >= 3) |>
  ggplot(aes(x = log10_mean_c57, y = log10_mean_nyob)) +
  geom_point(alpha = 0.4, size = 0.8, color = "steelblue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey40") +
  labs(
    title = "Per-species mean abundance: C57 vs NYOB",
    subtitle = "Points above diagonal = higher in NYOB; below = higher in C57",
    x = "log10(mean) in C57 samples",
    y = "log10(mean) in NYOB samples"
  ) +
  theme_bw(base_size = 11)

ggsave("plots/species_c57_vs_nyob.pdf", p_strain, width = 6, height = 6)
cat("Saved plots/species_c57_vs_nyob.pdf\n")

# ── PRE vs POST: within-mouse variance ────────────────────────────────────────

pre_ids  <- manifest |> filter(Time_Point == "PRE")  |> pull(Sample_ID)
post_ids <- manifest |> filter(Time_Point == "POST") |> pull(Sample_ID)

pre_mat  <- species_mat[, intersect(colnames(species_mat), pre_ids),  drop = FALSE]
post_mat <- species_mat[, intersect(colnames(species_mat), post_ids), drop = FALSE]

timepoint_stats <- tibble(
  name      = rownames(species_mat),
  mean_pre  = rowMeans(pre_mat),
  mean_post = rowMeans(post_mat),
  cv_pre    = apply(pre_mat,  1, sd) / (rowMeans(pre_mat)  + 1e-9),
  cv_post   = apply(post_mat, 1, sd) / (rowMeans(post_mat) + 1e-9),
) |>
  mutate(
    log2fc = log2((mean_post + 1e-9) / (mean_pre + 1e-9))
  ) |>
  filter(rowMeans(pre_mat) > 1e-6 | rowMeans(post_mat) > 1e-6)

# Volcano-style: log2FC (POST/PRE) vs mean abundance
p_volcano <- timepoint_stats |>
  ggplot(aes(x = log2fc, y = log10((mean_pre + mean_post)/2 + 1e-9))) +
  geom_point(aes(color = abs(log2fc) > 1), alpha = 0.5, size = 0.8) +
  scale_color_manual(values = c("grey60", "firebrick"), guide = "none") +
  geom_vline(xintercept = c(-1, 1), linetype = "dashed", color = "grey40") +
  labs(
    title = "Species-level PRE vs POST shift",
    subtitle = "Red = |log2FC| > 1 (more than 2x change after rewilding)",
    x = "log2(POST / PRE mean abundance)",
    y = "log10(mean abundance)"
  ) +
  theme_bw(base_size = 11)

ggsave("plots/species_pre_vs_post_volcano.pdf", p_volcano, width = 7, height = 5)
cat("Saved plots/species_pre_vs_post_volcano.pdf\n")

# ── Summary: how many species fall into each regime? ─────────────────────────

# Define regimes on detected species (prev >= 5)
det <- filter(species_stats, prevalence >= 5)

# Use log10 mean quantiles to define abundance regimes
low_thresh  <- quantile(det$log10_mean, 0.33)
high_thresh <- quantile(det$log10_mean, 0.67)

regime_stats <- det |>
  mutate(regime = case_when(
    log10_mean < low_thresh  ~ "low_abund",
    log10_mean > high_thresh ~ "high_abund",
    TRUE                     ~ "mid_abund"
  )) |>
  group_by(regime) |>
  summarise(
    n        = n(),
    mean_cv  = mean(cv),
    median_cv = median(cv),
    .groups = "drop"
  )

cat("\nSpecies by abundance regime (detected in >=5 samples):\n")
print(regime_stats)
cat("\nThree-regime hypothesis: mid-abundance species should have HIGHEST CV\n")
cat("(they vary most across samples -> most useful for differential coverage)\n")

# ── Save summary table ────────────────────────────────────────────────────────

write_tsv(species_stats, "gtdb_species_stats.tsv")
cat("\nFull species stats -> gtdb_species_stats.tsv\n")
