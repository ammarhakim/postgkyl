
from __future__ import annotations

import re


_LATEX_TO_UNICODE = {
    r"\mu": "μ",
    r"\nu": "ν",
    r"\pi": "π",
    r"\sigma": "σ",
    r"\Sigma": "Σ",
    r"\rho": "ρ",
    r"\tau": "τ",
    r"\chi": "χ",
    r"\phi": "φ",
    r"\psi": "ψ",
    r"\omega": "ω",
    r"\Omega": "Ω",
    r"\alpha": "α",
    r"\beta": "β",
    r"\gamma": "γ",
    r"\delta": "δ",
    r"\Delta": "Δ",
    r"\epsilon": "ε",
    r"\zeta": "ζ",
    r"\eta": "η",
    r"\theta": "θ",
    r"\Theta": "Θ",
    r"\iota": "ι",
    r"\kappa": "κ",
    r"\lambda": "λ",
    r"\Lambda": "Λ",
    r"\parallel": "∥",
    r"\perp": "⊥",
}


def latex_to_unicode(text: str) -> str:
  """Convert common LaTeX commands to Unicode."""
  if not text:
    return text
  # end
  text = text.strip()
  if text.startswith("$") and text.endswith("$"):
    text = text[1:-1]
  # end
  for latex, unicode_char in _LATEX_TO_UNICODE.items():
    text = text.replace(latex, unicode_char)
  # end
  return text


def latex_to_html(text: str) -> str:
  """Convert LaTeX subscripts and Greek letters to HTML.

  Plotly does not support LaTeX, but does support HTML, so this function
  converts common LaTeX syntax to HTML equivalents.
  """
  if not text:
    return text
  # end

  text = text.strip()
  if text.startswith("$") and text.endswith("$"):
    text = text[1:-1]
  # end

  def _replace_latex_commands(value: str) -> str:
    return latex_to_unicode(value)

  text = re.sub(
      r'_\{([^{}]+)\}',
      lambda match: f"<sub>{_replace_latex_commands(match.group(1))}</sub>",
      text,
  )
  text = re.sub(
      r'_(\\[A-Za-z]+|[A-Za-z0-9])',
      lambda match: f"<sub>{_replace_latex_commands(match.group(1))}</sub>",
      text,
  )
  text = _replace_latex_commands(text)
  return text


__all__ = ["latex_to_html", "latex_to_unicode"]