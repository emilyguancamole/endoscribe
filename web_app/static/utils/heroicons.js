// Heroicons helper - dynamically loads and inlines SVG icons from unpkg
const HeroIcon = {
    baseUrl: 'https://unpkg.com/heroicons@2.2.0',
    cache: new Map(),

    // Load an icon and return the SVG element
    async load(name, options = {}) {
        const size = options.size || 20;
        const style = options.style || 'solid';
        const className = options.class || 'h-5 w-5';

        const url = `${this.baseUrl}/${size}/${style}/${name}.svg`;

        // Check cache first
        if (this.cache.has(url)) {
            return this.createSvgElement(this.cache.get(url), className);
        }

        // Fetch from unpkg
        try {
            const response = await fetch(url);
            const svgText = await response.text();
            this.cache.set(url, svgText);
            return this.createSvgElement(svgText, className);
        } catch (error) {
            console.error(`Failed to load icon: ${name}`, error);
            return null;
        }
    },

    // Create SVG element from text and apply classes
    createSvgElement(svgText, className) {
        const parser = new DOMParser();
        const doc = parser.parseFromString(svgText, 'image/svg+xml');
        const svg = doc.querySelector('svg');
        if (svg && className) {
            svg.setAttribute('class', className);
        }
        return svg;
    },

    // Replace all elements with data-hero-icon attribute
    async replaceAll() {
        const elements = document.querySelectorAll('[data-hero-icon]');
        for (const el of elements) {
            const name = el.getAttribute('data-hero-icon');
            const size = el.getAttribute('data-hero-size') || 20;
            const style = el.getAttribute('data-hero-style') || 'solid';
            const className = el.getAttribute('data-hero-class') || el.className || 'h-5 w-5';

            const svg = await this.load(name, { size, style, class: className });
            if (svg) {
                el.replaceWith(svg);
            }
        }
    }
};

// Auto-replace icons on DOM load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => HeroIcon.replaceAll());
} else {
    HeroIcon.replaceAll();
}
