-- From wassname's unspoken-concepts: drop HTML comments (provenance, TODOs) from README.md.
-- README.qmd keeps them.
local function is_comment(el)
  return el.format == "html" and el.text:match("^%s*<!%-%-")
end
function RawBlock(el) if is_comment(el) then return {} end end
function RawInline(el) if is_comment(el) then return {} end end
