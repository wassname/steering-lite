import { defineConfig } from 'vite';

// relative base: the built page is opened next to its points.json (outputs/bsbench/results/<cohort>/)
export default defineConfig({ base: './', build: { emptyOutDir: false } });
