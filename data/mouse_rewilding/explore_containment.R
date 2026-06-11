library(tidyverse)
library(pheatmap)
library(RColorBrewer)

DATA_DIR <- "~/Documents/Projects/WF22_RefRover/RefRover/data/mouse_rewilding"

long    <- read_tsv(file.path(DATA_DIR, "containment_long.tsv"))
wide    <- read_tsv(file.path(DATA_DIR, "containment_wide.tsv"))
manifest <- read_tsv(file.path(DATA_DIR, "sample_manifest.tsv"))

# ── 1. Heatmap ────────────────────────────────────────────────────────────────

# Order samples: Trial → Timepoint → Subject
sample_order <- manifest |>
  arrange(Trial_ID, Time_Point, Subject_ID) |>
  pull(Sample_ID)

# Wide matrix: rows = queries (reads), cols = references (assemblies)
mat <- wide |>
  column_to_rownames("query_id") |>
  as.matrix()

# Reorder to match sample_order (some cols may be missing — m1082_POST assembly absent)
row_order <- intersect(sample_order, rownames(mat))
col_order <- intersect(sample_order, colnames(mat))
mat <- mat[row_order, col_order]

# Annotation bars for heatmap
ann_row <- manifest |>
  filter(Sample_ID %in% row_order) |>
  arrange(match(Sample_ID, row_order)) |>
  select(Sample_ID, Trial_ID, Time_Point, Mouse_Strain) |>
  column_to_rownames("Sample_ID")

ann_col <- manifest |>
  filter(Sample_ID %in% col_order) |>
  arrange(match(Sample_ID, col_order)) |>
  select(Sample_ID, Trial_ID, Time_Point) |>
  column_to_rownames("Sample_ID")

ann_colors <- list(
  Trial_ID    = setNames(brewer.pal(5, "Set1"), paste0("T00", 1:5)),
  Time_Point  = c(PRE = "#4393C3", POST = "#D6604D"),
  Mouse_Strain = c(C57 = "#984EA3", NYOB = "#FF7F00")
)

pdf(file.path(DATA_DIR, "heatmap_containment.pdf"), width = 14, height = 12)
pheatmap(
  mat,
  cluster_rows    = FALSE,
  cluster_cols    = FALSE,
  annotation_row  = ann_row,
  annotation_col  = ann_col,
  annotation_colors = ann_colors,
  color           = colorRampPalette(c("white", "#2166AC"))(100),
  breaks          = seq(0, 0.40, length.out = 101),
  show_rownames   = FALSE,
  show_colnames   = FALSE,
  main            = "Cross-sample containment: reads_i ∩ assembly_j / reads_i",
  fontsize        = 8
)
dev.off()
message("Saved: heatmap_containment.pdf")

# ── 2. μ and CV per reference assembly (the w_j signal) ──────────────────────

col_stats <- long |>
  group_by(reference_id) |>
  summarise(
    mean_c  = mean(containment),
    sd_c    = sd(containment),
    cv      = sd_c / mean_c,
    n       = n(),
    .groups = "drop"
  ) |>
  left_join(manifest |> select(Sample_ID, Trial_ID, Time_Point, Mouse_Strain),
            by = c("reference_id" = "Sample_ID"))

# Three-regime scatter
p_regimes <- ggplot(col_stats, aes(x = mean_c, y = cv, color = Time_Point, shape = Mouse_Strain)) +
  geom_point(size = 2.5, alpha = 0.8) +
  geom_vline(xintercept = c(0.05, 0.20), linetype = "dashed", color = "grey60") +
  annotate("text", x = 0.025, y = max(col_stats$cv, na.rm=TRUE)*0.95,
           label = "sparse\n(hard to recover)", hjust = 0.5, size = 3, color = "grey40") +
  annotate("text", x = 0.125, y = max(col_stats$cv, na.rm=TRUE)*0.95,
           label = "mid-abundance\nhigh variance\n← target", hjust = 0.5, size = 3, color = "#E41A1C") +
  annotate("text", x = 0.30, y = max(col_stats$cv, na.rm=TRUE)*0.95,
           label = "abundant / stable\n(recovers anyway)", hjust = 0.5, size = 3, color = "grey40") +
  scale_color_manual(values = c(PRE = "#4393C3", POST = "#D6604D")) +
  labs(
    x     = "Mean containment (μ_j)",
    y     = "CV of containment (CV_j = σ/μ)",
    title = "Per-assembly selection value signal",
    subtitle = "Mid-μ / high-CV assemblies are where prototype selection matters most"
  ) +
  theme_bw(base_size = 12)

ggsave(file.path(DATA_DIR, "regimes_scatter.pdf"), p_regimes, width = 8, height = 6)
message("Saved: regimes_scatter.pdf")

# Facet by trial to see if the regime distribution shifts across trials
p_by_trial <- p_regimes +
  facet_wrap(~Trial_ID, nrow = 2) +
  theme(legend.position = "bottom")
ggsave(file.path(DATA_DIR, "regimes_scatter_by_trial.pdf"), p_by_trial, width = 12, height = 8)
message("Saved: regimes_scatter_by_trial.pdf")

# ── 3. Within-mouse vs. cross-mouse hierarchy ─────────────────────────────────

meta_key <- manifest |> select(Sample_ID, Subject_ID, Time_Point, Trial_ID)

pairwise <- long |>
  left_join(meta_key, by = c("query_id"    = "Sample_ID")) |>
  rename(q_subject = Subject_ID, q_tp = Time_Point, q_trial = Trial_ID) |>
  left_join(meta_key, by = c("reference_id" = "Sample_ID")) |>
  rename(r_subject = Subject_ID, r_tp = Time_Point, r_trial = Trial_ID) |>
  filter(query_id != reference_id) |>
  mutate(
    pair_type = case_when(
      q_subject == r_subject                          ~ "Within-mouse",
      q_trial   == r_trial & q_subject != r_subject  ~ "Cross-mouse,\nsame trial",
      TRUE                                            ~ "Cross-trial"
    ) |> factor(levels = c("Within-mouse", "Cross-mouse,\nsame trial", "Cross-trial")),
    query_label = paste0(q_tp, " reads"),
    ref_label   = paste0(r_tp, " assembly")
  )

p_hierarchy <- pairwise |>
  filter(q_tp == "PRE", r_tp == "POST") |>   # most interpretable contrast
  ggplot(aes(x = pair_type, y = containment, fill = pair_type)) +
  geom_violin(alpha = 0.7, draw_quantiles = 0.5) +
  geom_jitter(width = 0.15, size = 0.4, alpha = 0.3) +
  scale_fill_brewer(palette = "Set2") +
  labs(
    x     = NULL,
    y     = "Containment (PRE reads → POST assembly)",
    title = "Containment hierarchy: within-mouse > same-trial > cross-trial",
    subtitle = "Validates that the matrix captures biological structure"
  ) +
  theme_bw(base_size = 12) +
  theme(legend.position = "none")

ggsave(file.path(DATA_DIR, "hierarchy_violin.pdf"), p_hierarchy, width = 7, height = 5)
message("Saved: hierarchy_violin.pdf")

# ── 4. Summary table ─────────────────────────────────────────────────────────

summary_tbl <- col_stats |>
  arrange(desc(cv)) |>
  select(reference_id, Trial_ID, Time_Point, Mouse_Strain, mean_c, cv) |>
  rename(mean_containment = mean_c)

write_tsv(summary_tbl, file.path(DATA_DIR, "assembly_selection_values.tsv"))
message("Saved: assembly_selection_values.tsv")

cat("\n--- Top 10 highest-CV assemblies (best prototype candidates) ---\n")
print(head(summary_tbl, 10), n = 10)

cat("\n--- Bottom 10 lowest-CV assemblies (least discriminating) ---\n")
print(tail(summary_tbl, 10), n = 10)
