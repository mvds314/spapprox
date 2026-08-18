# Skills

Project-specific skills for Claude Code / GitHub Copilot CLI.

Each skill lives in its own subfolder containing a `SKILL.md` file with YAML
frontmatter (`name`, `description`) plus instructions, e.g.:

```
.claude/skills/
  my-skill-name/
    SKILL.md
```

Both Claude Code and GitHub Copilot CLI read skills from this folder, so
skills placed here are shared between the two tools.
