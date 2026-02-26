# Event System

GLITCHLAB uses a centralized Event Bus to decouple core operations from infrastructure concerns.

## Usage

### Subscribing to Events
```javascript
eventBus.subscribe('user.created', (data) => {
  console.log('User created:', data);
});
```

### Publishing Events
```javascript
eventBus.publish('user.created', { id: 1, name: 'Nova' });
```

## Guidelines
1. **Naming**: Use dot-notation for event names (`domain.action`).
2. **Performance**: Subscribers are executed synchronously. Do not perform heavy computation inside a subscriber without offloading it.