let saveSessionSoon = () => {};

export function configurePromptRuntime(options = {}) {
  saveSessionSoon = options.saveSessionSoon || saveSessionSoon;
}

export function requestSessionSave() {
  saveSessionSoon();
}

export { saveSessionSoon };
