export const AUTH_GATE_OPEN_APP_MESSAGE = "Uygulamayi acmak icin once hesap olusturup giris yapmalisiniz."
export const AUTH_GATE_ROUTE_MESSAGE = "Uygulamaya gecmek icin oturum acmaniz gerekiyor."

export function canAccessApp(authToken, authUser) {
  return Boolean(authToken && authUser)
}

export function getProtectedButtonStyle(baseStyle, isAllowed) {
  if (isAllowed) return baseStyle
  return { ...baseStyle, opacity: 0.62 }
}
