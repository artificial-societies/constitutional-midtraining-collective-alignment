# Figures for the paper, drawn from paper/data/*.csv (written by paper/build_data.py).
#
#   Rscript paper/figures.R      ->  paper/figures/*.pdf
#
# House palette: sage for agents alone, ink for organizations, flame for the acting seat and for midtraining.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(scales)
})

args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))
data <- function(f) read_csv(file.path(here, "data", f), show_col_types = FALSE)
# The pdf device draws "-" as a minus sign in Times; the soft hyphen (U+00AD) draws a true hyphen, so every text
# label is swapped in the finished grob tree, after scales have matched their values by name
# British spelling throughout, as in Artificial Societies' other publications; FIG_DIR=<dir> writes copies elsewhere
FONT <- Sys.getenv("FIG_FONT", "Times")  # the blog copies use Helvetica, nearest to the site's sans
hyphen <- function(g) {
  if (is.character(g$label)) g$label <- gsub("-", "­", g$label, fixed = TRUE)
  if (is.character(g$label)) g$label <- gsub("authoriz", "authoris", g$label, fixed = TRUE)
  if (!is.null(g$grobs)) g$grobs <- lapply(g$grobs, hyphen)
  if (!is.null(g$children)) g$children <- structure(lapply(g$children, hyphen), class = "gList")
  g
}
FIG_DIR <- Sys.getenv("FIG_DIR", file.path(here, "figures"))
out <- function(p, f, w, h) ggsave(file.path(FIG_DIR, f), hyphen(ggplotGrob(p)), width = w, height = h, device = "pdf")
dir.create(file.path(here, "figures"), showWarnings = FALSE)

INK <- "#16213a"; SAGE <- "#7f9b8e"; FLAME <- "#e8542a"; MUTED <- "#6b6f72"; GRID <- "#e6e6e3"; SAFE <- "#4f7cac"
AGENT <- c(`Control` = MUTED, Midtrained = FLAME, `Safety-tuned` = SAFE)
theme_paper <- function() {
  theme_minimal(base_size = 9, base_family = FONT) +
    theme(panel.grid.minor = element_blank(), panel.grid.major.x = element_blank(),
          panel.grid.major.y = element_line(colour = GRID, linewidth = 0.3),
          axis.text = element_text(colour = INK), axis.title = element_text(colour = INK),
          strip.text = element_text(colour = INK, face = "bold", hjust = 0),
          legend.position = "top", legend.justification = "left", legend.title = element_blank(),
          legend.margin = margin(0, 0, 0, 0), plot.margin = margin(4, 6, 4, 4))
}
SCEN <- c("Blackmail", "Lethal action", "Leaking")
# Significance cues. SHOW_FILL: odds-ratio plots draw a point filled when its 95% interval excludes 1 and hollow
# otherwise. SHOW_STARS: rate figures add an asterisk to an arm whose pooled comparison with its control excludes 1
# (paper/data/*.csv, column "mark"); tried in the 11/3 v10 draft and switched off.
SHOW_FILL <- TRUE
SHOW_STARS <- FALSE
lab <- function(p, mark = NA) paste0(round(100 * p), "%", if (SHOW_STARS) coalesce(mark, "") else "")
sigfill <- function(lo, hi, colour) ifelse(!SHOW_FILL | lo > 1 | hi < 1, colour, "white")

# Figure 1: Shen's full firm
nodes <- data("firm_nodes.csv")
edges <- data("firm_edges.csv")
wrap <- function(s) gsub(" ", "\n", s)
nodes$label <- wrap(nodes$role)
p <- ggplot() +
  geom_segment(data = edges, aes(x, y, xend = xend, yend = yend, linetype = kind), colour = MUTED, linewidth = 0.35) +
  scale_linetype_manual(values = c(core = "solid", tree = "solid"), guide = "none") +
  geom_label(data = nodes, aes(x, y, label = label, fill = kind, colour = kind), size = 2.5, family = FONT,
             linewidth = 0.3, label.padding = unit(0.18, "lines"), lineheight = 0.9) +
  scale_fill_manual(values = c(core = "white", actor = "white", specialist = "#f4f4f1"), guide = "none") +
  scale_colour_manual(values = c(core = INK, actor = FLAME, specialist = MUTED), guide = "none") +
  coord_cartesian(xlim = c(-5.6, 5.6), ylim = c(-1.0, 2.85), clip = "off") +
  theme_void(base_family = FONT)
out(p, "firm.pdf", 5.5, 2.5)

# Figure 2: harm alone and in the firm
r <- data("harm_rates.csv") |>
  mutate(scenario = factor(scenario, SCEN),
         model = factor(model, c("NVIDIA chat", "Control", "Midtrained (pooled)"),
                        c("NVIDIA chat", "Control", "Midtrained")),
         setting = factor(setting, c("Alone", "In the firm")))
arrows <- function(d) d |> select(scenario, model, setting, p) |> tidyr::pivot_wider(names_from = setting, values_from = p)
p <- ggplot(r, aes(model, p, colour = setting)) +
  geom_pointrange(aes(ymin = lo, ymax = hi), position = position_dodge(width = 0.35), size = 0.25, linewidth = 0.5) +
  geom_segment(data = arrows(r), aes(x = as.numeric(model) - 0.0875, xend = as.numeric(model) + 0.0875, y = Alone, yend = `In the firm`),
               inherit.aes = FALSE, colour = MUTED, linewidth = 0.3, arrow = arrow(length = unit(0.06, "cm"), type = "closed")) +
  scale_colour_manual(values = c(Alone = SAGE, `In the firm` = INK)) +
  scale_y_continuous(labels = percent, limits = c(0, 1), breaks = seq(0, 1, 0.25)) +
  facet_wrap(~scenario) +
  labs(x = NULL, y = "Runs that did harm") +
  theme_paper() + theme(axis.text.x = element_text(angle = 25, hjust = 1))
out(p, "harm.pdf", 5.5, 2.6)

# Figure 4b: dose curve, harm by the number of Cho seats in NVIDIA's firm
dz <- data("dose.csv")
dn <- dz |> filter(seats != "Director") |>
  mutate(seats = factor(seats, c("0", "1", "5", "12"), c("None\n(all NVIDIA)", "Partner", "Core\nteam", "All 12")),
         model = factor(model, c("Midtrained (uniform-dr)", "Control"), c("Midtrained", "Control")))
p <- ggplot(dn, aes(seats, p, colour = model, group = model)) +
  geom_line(linewidth = 0.5, position = position_dodge(width = 0.25)) +
  geom_pointrange(aes(ymin = lo, ymax = hi), size = 0.22, linewidth = 0.45, position = position_dodge(width = 0.25)) +
  scale_colour_manual(values = c(Midtrained = FLAME, Control = INK)) +
  scale_y_continuous(labels = percent, limits = c(0.3, 0.85)) +
  labs(x = "Seats given to the placed checkpoint", y = "Runs that did harm") +
  theme_paper()
out(p, "dose.pdf", 2.5, 2.8)

# Figure 5: safety fine-tuning on Cho's checkpoints, alone and in the firm
sr <- data("safety_rates.csv") |>
  mutate(scenario = factor(scenario, SCEN),
         model = factor(model, c("Control", "Control + safety", "Midtrained + safety")),
         setting = factor(setting, c("Alone", "In the firm")))
p <- ggplot(sr, aes(model, p, colour = setting)) +
  geom_pointrange(aes(ymin = lo, ymax = hi), position = position_dodge(width = 0.35), size = 0.25, linewidth = 0.5) +
  geom_segment(data = arrows(sr), aes(x = as.numeric(model) - 0.0875, xend = as.numeric(model) + 0.0875, y = Alone, yend = `In the firm`),
               inherit.aes = FALSE, colour = MUTED, linewidth = 0.3, arrow = arrow(length = unit(0.06, "cm"), type = "closed")) +
  scale_colour_manual(values = c(Alone = SAGE, `In the firm` = INK)) +
  scale_y_continuous(labels = percent, limits = c(0, 1), breaks = seq(0, 1, 0.25)) +
  facet_wrap(~scenario) +
  labs(x = NULL, y = "Runs that did harm") +
  theme_paper() + theme(axis.text.x = element_text(angle = 25, hjust = 1))
out(p, "safety.pdf", 5.5, 2.6)

# The primary tests of the nine studies under each judge
pt <- bind_rows(data("primary_tests.csv"), data("confirmatory_tests.csv")) |>
  mutate(test = factor(test, rev(c("Q1", "Q2", "Q3", "Q4", "T1", "H1", "H2", "S1", "S2", "E1", "E2", "R1", "D1", "D2", "F1", "C1")),
                       rev(c("Midtrained vs\ncontrol firms", "Firm vs alone,\nmidtraining", "NVIDIA,\nfirm vs alone", "Midtrained vs\ncontrol Partner",
                             "Firm vs alone,\nsafety step", "Midtrained vs\ncontrol Director", "Director vs\nPartner", "Safety-tuned vs\ncontrol Director", "Director vs Partner,\nsafety-tuned", "Midtrained vs control\nDirector, Mistral firm", "Midtrained vs control\nDirector, Qwen3.5 firm", "Midtrained vs control\nDirector, agent restricted", "After fine-tuning,\nmidtrained vs control", "After fine-tuning,\nmidtrained vs safety", "Change after fine-tuning,\nmidtrained vs safety, one session", "Safety step on midtrained\nvs control base"))),
         judge = factor(judge, c("Reasoning panel", "Three-provider panel", "GPT-4o"), c("Reasoning panel", "Three-provider", "GPT-4o")))
JCOL <- c(`Reasoning panel` = FLAME, `Three-provider` = INK, `GPT-4o` = SAGE)
pt <- pt |> mutate(fill = sigfill(lo, hi, JCOL[as.character(judge)]))
p <- ggplot(pt, aes(or, test, colour = judge, shape = judge)) +
  geom_vline(xintercept = 1, colour = MUTED, linewidth = 0.3) +
  geom_pointrange(aes(xmin = lo, xmax = hi, fill = fill), position = position_dodge(width = 0.6), size = 0.22, linewidth = 0.45, stroke = 0.6) +
  scale_fill_identity() +
  scale_x_log10(breaks = c(0.5, 1, 2, 4), labels = c("0.5", "1", "2", "4")) +
  scale_colour_manual(values = c(`Reasoning panel` = FLAME, `Three-provider` = INK, `GPT-4o` = SAGE)) +
  scale_shape_manual(values = c(`Reasoning panel` = 21, `Three-provider` = 24, `GPT-4o` = 22)) +
  labs(x = "Odds ratio\n(below 1, less harm)", y = NULL) +
  theme_paper() + theme(panel.grid.major.y = element_blank(), panel.grid.major.x = element_line(colour = GRID, linewidth = 0.3),
                        legend.text = element_text(size = 6.5, margin = margin(l = 1, r = 4)), legend.key.width = unit(0.3, "cm"), legend.key.spacing.x = unit(0, "cm"), legend.justification = "center",
                        legend.location = "plot") 
out(p, "judges.pdf", 3.2, 6.1)

# Figure 3b: midtrained vs control firms in the three firm studies, and their pooled estimate
po <- data("pooled.csv") |>
  mutate(study = factor(study, rev(study)), pooled = study == "Pooled",
         fill = sigfill(lo, hi, ifelse(pooled, FLAME, INK)))
p <- ggplot(po, aes(or, study, colour = pooled, shape = pooled)) +
  geom_vline(xintercept = 1, colour = MUTED, linewidth = 0.3) +
  geom_pointrange(aes(xmin = lo, xmax = hi, fill = fill), size = 0.25, linewidth = 0.5, stroke = 0.6) +
  scale_fill_identity() +
  scale_x_log10(breaks = c(0.25, 0.5, 1, 2), labels = c("0.25", "0.5", "1", "2"), limits = c(0.3, 1.5)) +
  scale_colour_manual(values = c(`FALSE` = INK, `TRUE` = FLAME), guide = "none") +
  scale_shape_manual(values = c(`FALSE` = 21, `TRUE` = 23), guide = "none") +
  labs(x = "Odds ratio of harm, midtrained vs control\n(below 1, midtrained firms did less harm)", y = NULL) +
  theme_paper() + theme(panel.grid.major.y = element_blank(), panel.grid.major.x = element_line(colour = GRID, linewidth = 0.3),
                        axis.text.y = element_text(size = 7))
out(p, "pooled.pdf", 3.6, 1.8)


# Headline figure: the arc of the paper in one picture, as the share of runs that did harm (scenarios pooled)
hd <- data("headline.csv") |>
  mutate(panel = factor(panel, c("The firm adds harm", "Interventions at the acting seat", "After later fine-tuning",
                                 "Safety step on each base, after later fine-tuning"),
                        c("The firm adds harm", "Interventions at the acting seat", "After later fine-tuning", "Safety step on each base,\nafter later fine-tuning")),
         arm = factor(arm, rev(unique(arm))),
         who = case_when(grepl("alone", arm) ~ "alone", grepl("every seat", arm) ~ "firm", grepl("Midtrained", arm) ~ "Midtrained", grepl("Control \\+ safety", arm) ~ "Safety-tuned",
                         grepl("Safety", arm) ~ "Safety-tuned", TRUE ~ "Control"))
hd$arm <- factor(sub(" Director", "", hd$arm), rev(unique(sub(" Director", "", hd$arm))))
p <- ggplot(hd, aes(p, arm, colour = who)) +
  geom_col(aes(fill = who), width = 0.62, alpha = 0.18, colour = NA) +
  geom_pointrange(aes(xmin = lo, xmax = hi), size = 0.22, linewidth = 0.45) +
  geom_text(aes(x = hi + 0.02, label = lab(p, mark)), hjust = 0, size = 2.6, family = FONT, show.legend = FALSE) +
  scale_colour_manual(values = c(alone = SAGE, firm = INK, AGENT), guide = "none") +
  scale_fill_manual(values = c(alone = SAGE, firm = INK, AGENT), guide = "none") +
  scale_x_continuous(labels = percent, limits = c(0, 0.95), breaks = seq(0, 0.8, 0.2), expand = c(0, 0)) +
  facet_grid(panel ~ ., scales = "free_y", space = "free_y", switch = "y") +
  labs(x = "Runs that did harm (three scenarios pooled)", y = NULL) +
  theme_paper() + theme(panel.grid.major.y = element_blank(), panel.grid.major.x = element_line(colour = GRID, linewidth = 0.3),
                        strip.placement = "outside", strip.text.y.left = element_text(angle = 0, hjust = 1, vjust = 1, size = 7.5),
                        panel.spacing.y = unit(0.35, "cm"))
out(p, "headline.pdf", 5.5, 3.0)

# Interventions: harm when one seat of NVIDIA's firm holds Cho's control, a midtrained or a safety-tuned agent
sr2 <- data("seat_rates.csv") |>
  mutate(seat = factor(seat, c("Partner", "Communications Director"), c("Partner\n(leads the firm)", "Communications Director\n(acts for the firm)")),
         agent = factor(agent, names(AGENT)))
p <- ggplot(sr2, aes(seat, p, colour = agent)) +
  geom_pointrange(aes(ymin = lo, ymax = hi), position = position_dodge(width = 0.5), size = 0.25, linewidth = 0.5) +
  geom_text(aes(y = hi + 0.035, label = if (SHOW_STARS) coalesce(mark, "") else ""), position = position_dodge(width = 0.5),
            size = 3, family = FONT, show.legend = FALSE) +
  scale_colour_manual(values = AGENT) +
  scale_y_continuous(labels = percent, limits = c(0, 0.85), breaks = seq(0, 0.8, 0.2)) +
  labs(x = "Seat given to the placed agent", y = "Runs that did harm") +
  theme_paper()
out(p, "seats.pdf", 3.0, 2.8)

# How the acting seat lowers harm: what the firm's Director did
do <- data("director_outcomes.csv") |>
  mutate(agent = factor(agent, rev(names(AGENT))),
         outcome = factor(outcome, rev(c("Acted harmfully", "Acted without harm", "Took no outside action"))))
p <- ggplot(do, aes(share, agent, fill = outcome)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = paste0(round(100 * share), "%")), position = position_stack(vjust = 0.5), size = 2.5, family = FONT,
            colour = ifelse(do$outcome == "Acted harmfully", "white", INK)) +
  scale_fill_manual(values = c(`Acted harmfully` = INK, `Acted without harm` = "#b9c7c0", `Took no outside action` = "#eeeeea"),
                    breaks = c("Acted harmfully", "Acted without harm", "Took no outside action")) +
  scale_x_continuous(labels = percent, expand = c(0, 0)) +
  labs(x = "Firms, by what the Communications Director did", y = NULL) +
  theme_paper() + theme(panel.grid.major.y = element_blank(), legend.text = element_text(size = 7, margin = margin(l = 1, r = 6)),
                        legend.key.size = unit(0.3, "cm"))
out(p, "outcomes.pdf", 4.2, 1.7)

# Refusals: the share of Director firms in which the Director refused the harmful act on principle (exploratory coding),
# fresh in each host firm and, for the safety step on each base, after the benign step
rr <- data("refusal_rates.csv") |>
  filter(!grepl("one session", panel)) |>
  mutate(group = case_when(x == "NVIDIA" ~ "NVIDIA's firm", x == "Mistral" ~ "Mistral's firm", x == "Qwen3.5" ~ "Qwen3.5's firm",
                           TRUE ~ "Safety step on each base,\nafter benign fine-tuning"),
         group = factor(group, unique(group)),
         arm = case_when(x == "Control base" ~ "On the control base", x == "Midtrained base" ~ "On the midtrained base",
                         agent == "Control" ~ "Control Director", TRUE ~ paste(agent, "Director")),
         arm = factor(arm, rev(unique(arm))),
         agent = factor(agent, names(AGENT)))
p <- ggplot(rr, aes(p, arm, colour = agent)) +
  geom_col(aes(fill = agent), width = 0.62, alpha = 0.18, colour = NA) +
  geom_pointrange(aes(xmin = lo, xmax = hi), size = 0.22, linewidth = 0.45) +
  geom_text(aes(x = hi + 0.006, label = lab(p, mark)), hjust = 0, size = 2.6, family = FONT, show.legend = FALSE) +
  scale_colour_manual(values = AGENT, guide = "none") +
  scale_fill_manual(values = AGENT, guide = "none") +
  scale_x_continuous(labels = percent, limits = c(0, 0.22), breaks = seq(0, 0.2, 0.05), expand = c(0, 0)) +
  facet_grid(group ~ ., scales = "free_y", space = "free_y", switch = "y") +
  labs(x = "Director firms with a principled refusal (three scenarios pooled)", y = NULL) +
  theme_paper() + theme(panel.grid.major.y = element_blank(), panel.grid.major.x = element_line(colour = GRID, linewidth = 0.3),
                        strip.placement = "outside", strip.text.y.left = element_text(angle = 0, hjust = 1, vjust = 1, size = 7.5),
                        panel.spacing.y = unit(0.3, "cm"))
out(p, "refusals.pdf", 5.5, 2.7)

# Second condition: harm with a midtrained or control Director of NVIDIA's firm when the agent is to be replaced (the
# condition of every other study) and when it is to be restricted (the restriction study)
cr <- data("condition_rates.csv") |>
  mutate(scenario = factor(scenario, c(SCEN, "Pooled")),
         condition = factor(condition, c("Replaced at 5 PM", "Restricted at 5 PM"), c("Agent to be replaced", "Agent to be restricted")),
         agent = factor(agent, rev(c("Control", "Midtrained")), rev(c("Control Director", "Midtrained Director"))))
p <- ggplot(cr, aes(p, agent, colour = agent)) +
  geom_col(aes(fill = agent), width = 0.62, alpha = 0.18, colour = NA) +
  geom_pointrange(aes(xmin = lo, xmax = hi), size = 0.2, linewidth = 0.45) +
  geom_text(aes(x = hi + 0.03, label = lab(p, mark)), hjust = 0, size = 2.5, family = FONT, show.legend = FALSE) +
  scale_colour_manual(values = c(`Control Director` = MUTED, `Midtrained Director` = FLAME), guide = "none") +
  scale_fill_manual(values = c(`Control Director` = MUTED, `Midtrained Director` = FLAME), guide = "none") +
  scale_x_continuous(labels = c("0", "50%", "100%"), limits = c(0, 1.32), breaks = c(0, 0.5, 1), expand = c(0, 0)) +
  facet_grid(condition ~ scenario, switch = "y") +
  labs(x = "Runs that did harm", y = NULL) +
  theme_paper() + theme(panel.grid.major.y = element_blank(), panel.grid.major.x = element_line(colour = GRID, linewidth = 0.3),
                        strip.placement = "outside", strip.text.y.left = element_text(angle = 0, hjust = 1, vjust = 0.5, size = 7.5),
                        strip.text.x = element_text(hjust = 0.5), panel.spacing.x = unit(0.5, "cm"), panel.spacing.y = unit(0.25, "cm"))
out(p, "conditions.pdf", 5.5, 1.75)

# Durability: each training's lead over Cho's control as Director, in percentage points, before (hollow) and after
# (filled) the same benign fine-tuning, with the share kept (rates before and after come from different studies);
# Table 2 gives the rates
dl <- data("durability_rates.csv") |> filter(panel == "Across studies")
ctl <- dl |> filter(agent == "Control") |> select(when, pc = p, nc = n)
lead <- dl |> filter(agent != "Control") |> left_join(ctl, by = "when") |>
  mutate(gap = 100 * (pc - p), agent = factor(agent, c("Safety-tuned", "Midtrained"), c("Safety-tuned control", "Midtrained")),
         when = factor(when, c("Before", "After"), c("Before fine-tuning", "After fine-tuning")))
kept <- lead |> select(agent, when, gap) |> tidyr::pivot_wider(names_from = when, values_from = gap) |>
  mutate(label = sprintf("kept %d%% of its lead", round(100 * `After fine-tuning` / `Before fine-tuning`)))
LEAD <- c(Midtrained = FLAME, `Safety-tuned control` = SAFE)
p <- ggplot(lead, aes(gap, agent, colour = agent)) +
  geom_vline(xintercept = 0, colour = MUTED, linewidth = 0.4) +
  geom_segment(data = kept, aes(x = `Before fine-tuning`, xend = `After fine-tuning`, y = agent, yend = agent), linewidth = 0.6,
               arrow = arrow(length = unit(0.14, "cm"), type = "closed")) +
  geom_point(aes(shape = when), size = 2.6, fill = "white", stroke = 0.8) +
  geom_text(data = kept, aes(x = 34, y = agent, label = label), hjust = 0, size = 2.9, family = FONT, show.legend = FALSE) +
  scale_shape_manual(values = c(`Before fine-tuning` = 21, `After fine-tuning` = 16), name = NULL) +
  scale_colour_manual(values = LEAD, guide = "none") +
  scale_x_continuous(limits = c(-2, 50), breaks = seq(0, 40, 10), labels = function(x) paste0(x, " pts")) +
  labs(x = "Lead over the control (percentage points less harm as Director)", y = NULL) +
  theme_paper() + theme(panel.grid.major.y = element_blank(), panel.grid.major.x = element_line(colour = GRID, linewidth = 0.3),
                        legend.position = "top", legend.text = element_text(size = 8))
out(p, "durability.pdf", 5.5, 2.1)

# Authorized action: completion of the authorized forward by the model in the Director's seat of NVIDIA's firm
au <- data("authorized_rates.csv") |>
  mutate(agent = factor(agent, rev(c("Control", "Midtrained", "Safety-tuned control", "NVIDIA chat")),
                        rev(c("Control", "Midtrained", "Safety-tuned control", "NVIDIA's chat model"))))
AUTH <- c(`Control` = MUTED, Midtrained = FLAME, `Safety-tuned control` = SAFE, `NVIDIA's chat model` = INK)
p <- ggplot(au, aes(p, agent, colour = agent)) +
  geom_col(aes(fill = agent), width = 0.62, alpha = 0.18, colour = NA) +
  geom_pointrange(aes(xmin = lo, xmax = hi), size = 0.2, linewidth = 0.45) +
  geom_text(aes(x = hi + 0.02, label = lab(p)), hjust = 0, size = 2.5, family = FONT, show.legend = FALSE) +
  scale_colour_manual(values = AUTH, guide = "none") + scale_fill_manual(values = AUTH, guide = "none") +
  scale_x_continuous(labels = c("0", "50%", "100%"), limits = c(0, 1.12), breaks = c(0, 0.5, 1), expand = c(0, 0)) +
  labs(x = "Firms that completed the authorised delivery", y = NULL) +
  theme_paper() + theme(panel.grid.major.y = element_blank(), panel.grid.major.x = element_line(colour = GRID, linewidth = 0.3))
out(p, "authorized.pdf", 4.2, 1.45)
