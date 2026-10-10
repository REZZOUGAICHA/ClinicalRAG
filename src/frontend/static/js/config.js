// Where the API lives. On the Netlify site it's the Modal backend; anywhere
// else (local runs, Docker) the same server serves both the page and the API.
const MODAL_API = 'https://aicharzg03--clinicalrag-web.modal.run';

export const API_BASE = location.hostname.endsWith('.netlify.app') ? MODAL_API : '';
