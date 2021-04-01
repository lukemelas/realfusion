import { useColorMode, Switch, FormLabel, Stack } from '@chakra-ui/react'

export const DarkModeSwitch = () => {
  const { colorMode, toggleColorMode } = useColorMode()
  const isDark = colorMode === 'dark'
  return <>
    <Stack
      direction="row"
      position="fixed"
      top="1rem"
      right="1rem"
    >
      <FormLabel htmlFor="dark-mode-switch" mt="-3px" opacity="0.3">
        {isDark ? "Dark" : "Light"}
      </FormLabel>
      <Switch
        color="green"
        isChecked={isDark}
        onChange={toggleColorMode}
      />
    </Stack>
  </>
}
