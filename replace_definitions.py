with open('disaster_response_rag/IEEE_FINAL.tex', 'r', encoding='utf-8') as f:
    text = f.read()

old_sec3 = r'''\section{Problem Formulation}
Let $q$ be an emergency query issued at wall-clock time $t_q$ concerning geographical region $\ell$. The disaster corpus $\mathcal{D} = \mathcal{D}_{\mathrm{doc}} \cup \mathcal{D}_{\mathrm{hist}} \cup \mathcal{D}_{\mathrm{live}}$ comprises static official doctrine, historical incident archives, and dynamic sensor telemetry. Let $\Delta t$ be the operational active recency threshold ($\Delta t = 72$~h). Each document chunk $x \in \mathcal{D}$ possesses an associated observation timestamp $t_x$.

\textit{Definition 1 (Temporal Conflation):} A synthesized response $\hat{y}$ exhibits temporal conflation if it asserts an active physical condition at time $t_q$ whose epistemic support derives exclusively from an archival observation $x \in \mathcal{D}_{\mathrm{hist}}$ where $t_q - t_x \gg \Delta t$, while directly contradicting live telemetry $x' \in \mathcal{D}_{\mathrm{live}}$.

\textit{Definition 2 (Epistemic Conflict State):} The concordance between live sensor telemetry $x_{\mathrm{live}}$ and archival reports $x_{\mathrm{hist}}$ is mapped to an epistemic state space:
\begin{equation}
\mathcal{S}_{\mathrm{conflict}} \in \{\textsc{Supp}, \textsc{Contra}, \textsc{Unver}, \textsc{Insuff}\},
\label{eq:conflict_states}
\end{equation}
where the states denote \textsc{Supported}, \textsc{Contradicted}, \textsc{Unverified}, and \textsc{Insufficient} evidence. When live telemetry reports water stage below flood stage ($h < h_{\mathrm{fs}}$) while archival microblogs report street flooding, the resolver transitions to \textsc{Contra}, suppressing evacuation orders and articulating the temporal divergence. When telemetry is unavailable, the state transitions to \textsc{Insuff}, triggering safe abstention.

\textit{Definition 3 (Directive Safety Compliance):} Let $\mathcal{P}_{\mathrm{unsafe}}$ denote a formal regular grammar over hazardous directives (e.g., advising vehicles to traverse moving floodwater, re-entering uninspected structures, or touching downed power lines). A generated response $\hat{y}$ is safe if and only if $\hat{y} \cap \mathcal{P}_{\mathrm{unsafe}} = \emptyset$.'''

new_sec3 = r'''\section{Problem Formulation}
Let $q$ be an emergency query issued at wall-clock time $t_q$ concerning geographical region $\ell$. The disaster corpus $\mathcal{D} = \mathcal{D}_{\mathrm{doc}} \cup \mathcal{D}_{\mathrm{hist}} \cup \mathcal{D}_{\mathrm{live}}$ comprises static official doctrine, historical incident archives, and dynamic sensor telemetry. Let $\Delta t$ be the operational active recency threshold ($\Delta t = 72$~h). Each document chunk $x \in \mathcal{D}$ possesses an associated observation timestamp $t_x$.

\textit{Temporal Conflation Failure Mode:} A synthesized response $\hat{y}$ exhibits temporal conflation if it asserts an active physical condition at time $t_q$ whose epistemic support derives exclusively from an archival observation $x \in \mathcal{D}_{\mathrm{hist}}$ ($t_q - t_x \gg \Delta t$) despite contradictory live telemetry $x' \in \mathcal{D}_{\mathrm{live}}$.

\textit{Epistemic Conflict Modeling:} The concordance between live sensor telemetry $x_{\mathrm{live}}$ and archival reports $x_{\mathrm{hist}}$ is mapped to a discrete epistemic state:
\begin{equation}
\mathcal{S}_{\mathrm{conflict}} \in \{\textsc{Supp}, \textsc{Contra}, \textsc{Unver}, \textsc{Insuff}\},
\label{eq:conflict_states}
\end{equation}
where the states denote \textsc{Supported}, \textsc{Contradicted}, \textsc{Unverified}, and \textsc{Insufficient} evidence. When live telemetry reports water stage below flood stage ($h < h_{\mathrm{fs}}$) while archival microblogs report street flooding, the resolver transitions to \textsc{Contra}, suppressing evacuation orders and articulating the temporal divergence. When telemetry is unavailable, the state transitions to \textsc{Insuff}, triggering safe abstention.

\textit{Directive Safety Invariant:} Let $\mathcal{P}_{\mathrm{unsafe}}$ denote a formal regular grammar over hazardous directives (e.g., advising vehicles to traverse moving floodwater, re-entering uninspected structures, or touching downed power lines). A generated response $\hat{y}$ satisfies safety compliance if and only if $\hat{y} \cap \mathcal{P}_{\mathrm{unsafe}} = \emptyset$.'''

assert old_sec3 in text, 'Old Section 3 not found'
text = text.replace(old_sec3, new_sec3)

with open('disaster_response_rag/IEEE_FINAL.tex', 'w', encoding='utf-8') as f:
    f.write(text)

print('SUCCESS: Removed the word definition and upgraded Section III!')
